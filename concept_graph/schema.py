"""Typed graph schema. Every triple the LLM returns is validated against these models.

The schema is FIXED: the LLM may only use the entity and relation types below.
Anything else is rejected and logged (see extract.py); it is never silently accepted.
RELATED_TO is the only generic relation, kept so that a true fact that fits no other type
is not lost. It is reported separately in the build stats so its share can be watched.
"""
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    SERVICE = "Service"            # postroom, ledger, gateway
    TEAM = "Team"                  # Messaging, Platform, Payments Core
    PERSON = "Person"              # Samuel Okafor, Yakov Glinkin
    CHANNEL = "Channel"            # #oncall-messaging, an e-mail address
    ERROR_CODE = "ErrorCode"       # KST-5031
    PLAN = "Plan"                  # Business, Enterprise
    FEATURE = "Feature"            # scheduled reports, SSO, embed tokens
    POLICY = "Policy"              # grace period, refund rule, rate limit
    DOCUMENT = "Document"          # Billing and Refund Policy 2026
    PLACE = "Place"                # Tver, Stolyarny Lane
    OBJECT = "Object"              # silver notebook
    ORGANIZATION = "Organization"  # payment processor, customer company


class RelationType(str, Enum):
    OWNS = "OWNS"                  # Team -> Service
    LEADS = "LEADS"                # Person -> Team
    MEMBER_OF = "MEMBER_OF"        # Person -> Team / Organization
    CONTACT_VIA = "CONTACT_VIA"    # Team -> Channel
    RAISES = "RAISES"              # Service -> ErrorCode
    ESCALATE_TO = "ESCALATE_TO"    # ErrorCode / Team -> Team
    DEPENDS_ON = "DEPENDS_ON"      # Service -> Service
    SENDS = "SENDS"                # Service -> Feature (postroom SENDS scheduled reports)
    REPLACES = "REPLACES"          # new Service/Document -> old one
    INCLUDES = "INCLUDES"          # Plan -> Feature
    APPLIES_TO = "APPLIES_TO"      # Policy -> Plan / Feature
    DEFINED_IN = "DEFINED_IN"      # Policy -> Document
    LOCATED_IN = "LOCATED_IN"      # anything -> Place
    MOVED_TO = "MOVED_TO"          # Person / Object -> Place
    HAS = "HAS"                    # Person -> Object
    LENT_TO = "LENT_TO"            # Object -> Person (the person who received it)
    SAME_AS = "SAME_AS"            # alias: "the old bookbinder" SAME_AS "Yakov Glinkin"
    RELATED_TO = "RELATED_TO"      # generic fallback, counted separately


class EntityRef(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    type: EntityType

    @field_validator("name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("empty name")
        return v


class Triple(BaseModel):
    subject: EntityRef
    relation: RelationType
    object: EntityRef
    # Verbatim sentence fragment from the chunk that states the fact. It is checked against the
    # chunk text after validation: a quote that is not in the chunk makes the triple "unsupported".
    evidence: str = Field(min_length=5, max_length=400)


class Extraction(BaseModel):
    triples: list[Triple]


SCHEMA_DESCRIPTION = (
    "Entity types: " + ", ".join(t.value for t in EntityType) + "\n"
    "Relation types: " + ", ".join(r.value for r in RelationType)
)
