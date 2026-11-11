# The corpus

`corpus.jsonl` holds every article (the enacting terms, without recitals or annexes) of three EU acts, one JSON object per article: 239 articles, about 420,000 characters.

| Key | Act | Articles | CELEX |
|---|---|---|---|
| PSD2 | Directive (EU) 2015/2366 on payment services in the internal market | 117 | [32015L2366](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32015L2366) |
| DGSD | Directive 2014/49/EU on deposit guarantee schemes | 23 | [32014L0049](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32014L0049) |
| GDPR | Regulation (EU) 2016/679, the General Data Protection Regulation | 99 | [32016R0679](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679) |

`python -m bankrag build-corpus` downloads the English Official Journal text of each act from the EU Publications Office (`publications.europa.eu/resource/celex/<CELEX>`) and parses it with `bankrag/corpus.py`. EUR-Lex itself answers scripts with a bot check, so the Publications Office endpoint is used instead; it serves the same text.

These are the acts **as originally published**, without later amendments or corrigenda, and some have since been amended or are being replaced. The corpus is test material for a retrieval pipeline, not a statement of current law.

**Source and reuse:** © European Union, https://eur-lex.europa.eu, 1998–2026. Reuse of EUR-Lex content is authorised provided the source is acknowledged (Commission Decision 2011/833/EU). Only European Union legal documents published in the Official Journal are deemed authentic.
