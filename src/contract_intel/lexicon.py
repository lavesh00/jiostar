"""Small hand-written query-expansion lexicon for common contract-review questions.

Provenance: written from general contract-drafting knowledge and the public CUAD category names
(Hendrycks et al., CC BY 4.0) after looking at the *question templates* in the train split. It was NOT derived
from gold answer spans or contract text. Each rule maps trigger phrases found in the question to terms
that clauses of that kind typically contain; matching rules append their terms to the query.
Risk: the lexicon encodes CUAD-style categories, so it may help less on questions phrased differently.
"""

from __future__ import annotations

RULES: list[tuple[tuple[str, ...], str]] = [
    (("expiration date", "expire", "duration of"), "term commence effective date expire expiration years initial term unless terminated"),
    (("renewal term", "renew"), "renew renewal extended additional successive period automatically term"),
    (("notice period",), "notice days prior written non-renewal terminate renewal"),
    (("governing law", "governed"), "governed construed laws state courts jurisdiction accordance"),
    (("anti-assignment", "assign"), "assign assignment transfer consent written prior delegate successors"),
    (("cap on liability", "liability under"), "liability limited aggregate exceed cap consequential indirect damages shall not be liable"),
    (("uncapped",), "unlimited uncapped liability without limitation indemnify"),
    (("licenses are granted", "licenses granted", "license"), "grants grant license licensed non-exclusive exclusive right use"),
    (("non-transferable",), "non-transferable not transferable transfer assign sublicense"),
    (("irrevocable", "perpetual"), "irrevocable perpetual license royalty-free"),
    (("affiliates",), "affiliates affiliate subsidiaries license rights"),
    (("audit",), "audit inspect inspection records books accounts examine access"),
    (("competitive restrictions", "non-compete", "compete"), "compete competing competitive competition business restriction except"),
    (("intellectual property ownership", "intellectual property"), "intellectual property ownership owns title rights jointly developed work product assigns"),
    (("terminated for convenience", "for convenience"), "terminate convenience without cause notice days either party"),
    (("exclusivity", "exclusive"), "exclusive exclusively sole only exclusivity"),
    (("after the termination", "post-termination"), "after termination expiration continue provide transition services wind-down"),
    (("volume restrictions", "volume"), "volume quantity minimum maximum units purchase"),
    (("minimum commitment",), "minimum purchase commitment quantity annual"),
    (("insurance",), "insurance coverage maintain policy liability certificate"),
    (("revenue", "profit-sharing", "profit sharing"), "revenue profit share split percent net sales royalty fee"),
    (("right of first refusal", "first offer", "first negotiation"), "right of first refusal offer negotiation first"),
    (("solicitation of employees", "solicit"), "solicit hire employees personnel customers non-solicitation"),
    (("covenant not to sue",), "covenant not to sue challenge contest validity"),
    (("third-party beneficiar", "third party beneficiar"), "third party beneficiary beneficiaries rights enforce"),
    (("warranties", "warranty"), "warranty warranties warrants period months years defects"),
    (("change of control",), "change of control merger acquisition consolidation acquire voting"),
    (("most favored", "most favoured"), "most favored favoured nation lowest price terms no less favorable"),
]


def expand(question: str) -> str:
    """Append lexicon terms for every rule triggered by the question (deterministic, order of RULES)."""
    q = question.lower()
    extra = [terms for triggers, terms in RULES if any(t in q for t in triggers)]
    return question + (" " + " ".join(extra) if extra else "")
