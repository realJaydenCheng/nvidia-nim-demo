
import base64
import re
from io import BytesIO
from typing import Literal

import fitz
import requests
from fitz import Document
from openai import OpenAI
import numpy as np

from __env import NVIDIA_API_KEY


def extract_images_from_pdf(doc: Document):

    # see: https://github.com/pymupdf/PyMuPDF-Utilities/blob/master/examples/extract-images/extract-from-xref.py

    dimlimit = 100  # each image side must be greater than this
    relsize = 0.05  # image : pixmap size ratio must be larger than this (5%)
    abssize = 2048  # absolute image size limit 2 KB: ignore if smaller

    def recoverpix(doc, x, imgdict):
        """Return pixmap for item with an image mask."""
        s = imgdict["smask"]  # xref of its image mask

        try:
            pix0 = fitz.Pixmap(imgdict["image"])
            mask = fitz.Pixmap(doc.extract_image(s)["image"])
            pix = fitz.Pixmap(pix0, mask)
            if pix0.n > 3:
                ext = "pam"
            else:
                ext = "png"
            return {"ext": ext, "colorspace": pix.colorspace.n, "image": pix.tobytes(ext)}
        except:
            return None

    lenXREF = doc.xref_length()  # PDF object count - do not use entry 0!

    smasks = set()  # stores xrefs of /SMask objects
    # ------------------------------------------------------------------------------
    # loop through PDF images
    # ------------------------------------------------------------------------------
    out_imgs: list[BytesIO] = []
    for xref in range(1, lenXREF):  # scan through all PDF objects

        if doc.xref_get_key(xref, "Subtype")[1] != "/Image":  # not an image
            continue
        if xref in smasks:  # ignore smask
            continue

        imgdict = doc.extract_image(xref)

        if not imgdict:  # not an image
            continue

        smask = imgdict["smask"]
        if smask > 0:  # store /SMask xref
            smasks.add(smask)

        width = imgdict["width"]
        height = imgdict["height"]

        if min(width, height) <= dimlimit:  # rectangle edges too small
            continue

        imgdata = imgdict["image"]  # image data
        l_imgdata = len(imgdata)  # length of data
        if l_imgdata <= abssize:  # image too small to be relevant
            continue

        if smask > 0:  # has smask: need use pixmaps
            # create pix with mask applied
            imgdict = recoverpix(doc, xref, imgdict)
            if imgdict is None:  # something went wrong
                continue
            imgdata = imgdict["image"]
            l_samples = width * height * 3
            l_imgdata = len(imgdata)
        else:
            c_space = max(1, imgdict["colorspace"])  # get the colorspace n
            l_samples = width * height * c_space  # simulated samples size

        if l_imgdata / l_samples <= relsize:  # seems to be unicolor image
            continue

        out_imgs.append(BytesIO(imgdata))

    return out_imgs


def img_to_text(img: BytesIO):
    invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
    image_b64 = base64.b64encode(img.read()).decode()
    assert len(image_b64) < 180_000, \
        "To upload larger images, use the assets API (see docs)"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json"
    }
    payload = {
        "model": 'meta/llama-3.2-11b-vision-instruct',
        "messages": [
            {
                "role": "user",
                "content": f'Describe this img in one sentence: <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9,
        "stream": False,
    }
    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


def text_to_embd_one_batch(
    text: list[str],
    type: Literal["query", "passage"],
    client: OpenAI
):
    response = client.embeddings.create(
        input=text,
        model="nvidia/nv-embedqa-e5-v5",
        encoding_format="float",
        extra_body={"input_type": type, "truncate": "NONE"}
    )

    return [
        obj.embedding for obj in response.data
    ]


def text_to_embd(
    text: list[str],
    type: Literal["query", "passage"] = "passage",
    batch_size: int = 128,
):
    client = OpenAI(
        api_key=NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1"
    )

    result = []
    for i in range(0, len(text), batch_size):
        result.extend(text_to_embd_one_batch(
            text[i:i + batch_size], type, client
        ))
    return result


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
):
    clean_text = re.sub(r'[\s\n]+', ' ', text, flags=re.S).strip()
    chunked = [
        clean_text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size - overlap)
    ]
    return [
        chunk for chunk in chunked
        if re.sub(r'[\s\n]+', '', chunk, flags=re.S).strip()
    ]


def recall_passages(
    query: str,
    chunk_embds: np.ndarray,
    threshold: float,
    top_k: int,
):
    query_embd = text_to_embd([query], "query")
    similarities = np.dot(chunk_embds, query_embd[0])
    indices = np.array([])
    for t in range(int(threshold * 100), 0, -5):
        t = t / 100
        indices = np.where(similarities > t / threshold)[0]
        if len(indices) > top_k:
            break
    return indices


def rank_passages(
    query: str,
    passages: list[str],
):
    invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    payload = {
        "model": "nvidia/nv-rerankqa-mistral-4b-v3",
        "query": {"text": query},
        "passages": [
            {"text": passage}
            for passage in passages
        ]
    }

    # re-use connections
    session = requests.Session()
    response = session.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()

    rankings: list[int] = [
        data["index"] for data in response.json()['rankings']
    ]
    return rankings


class IngestedDoc:

    def __init__(self, pdf_content: BytesIO) -> None:
        self.input = pdf_content
        self.file = fitz.open(stream=self.input)
        self._parse_pdf()
        self._gen_full_text()
        self._chunk_and_embedding()

    def retrieve(
        self,
        query: str,
        top_k=5,
        max_threshold: float = 0.8,
    ):
        recalled_indices = recall_passages(
            query, self.embeddings,
            max_threshold, top_k
        )
        recalled_chunks = [self.chunked_text[i] for i in recalled_indices]
        ranked_indices = rank_passages(query, recalled_chunks)
        return [
            recalled_chunks[i]
            for i in ranked_indices[:top_k]
        ]

    def _chunk_and_embedding(self):
        self.chunked_text = chunk_text(self.full_text)
        self.embeddings = np.array(
            text_to_embd(self.chunked_text, "passage")
        )

    def _parse_pdf(self):
        self.text = "\n".join([
            page.get_text() for page in self.file
        ])
        self.images = extract_images_from_pdf(self.file)

    def _gen_full_text(self):
        img_text_ls = [
            img_to_text(img)
            for img in self.images
        ]
        self.full_text = self.text + "\n" + "\n".join(img_text_ls)


if __name__ == "__main__":
    # test scripts
    file = open("test.pdf", "rb")
    doc = IngestedDoc(BytesIO(file.read()))
    print(doc.full_text)
    print("\n\n" + "*" * 32 + "\n\n")
    [
        print(t) for t in
        doc.retrieve(
            "Retaining loyal users is a key factor in a startup's success."
        )
    ]
