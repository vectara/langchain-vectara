# langchain-vectara

This package contains the LangChain integration with Vectara.

## Installation

```bash
pip install -U langchain-vectara
```

And you should configure credentials by setting the following environment variables:

- `Vectara_API_KEY`

## Usage

The `Vectara` class exposes the functionality of Vectara.

```python
from langchain_vectara import Vectara

vectorstore = Vectara()
```
