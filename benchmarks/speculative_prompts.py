"""Workloads for comparing speculative-decoding draft-token policies."""

CODE_REFACTORING_PROMPT = r'''
Refactor the Python module below by renaming the `record` parameter to `payload` in every public
function and updating every reference to that parameter. Make no other code changes: do not
extract helpers, consolidate validation, rename local variables, reorder functions or fields, or
alter imports, annotations, comments, event names, error messages, result dictionaries, or
observable behavior. Return the complete refactored module, including imports and every function.
Output Python code only, without Markdown fences, explanations, ellipses, or omitted sections.

from collections.abc import Mapping
from typing import Any


def build_user_created(record: Mapping[str, Any]) -> dict[str, str]:
    # User lifecycle event consumed by the identity indexer.
    user_id = str(record.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("missing user_id")
    email = str(record.get("email", "")).strip()
    if not email:
        raise ValueError("missing email")
    region = str(record.get("region", "")).strip()
    if not region:
        raise ValueError("missing region")
    return {
        "event": "user.created",
        "user_id": user_id,
        "email": email,
        "region": region,
    }


def build_user_updated(record: Mapping[str, Any]) -> dict[str, str]:
    # User lifecycle event consumed by the identity indexer.
    user_id = str(record.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("missing user_id")
    email = str(record.get("email", "")).strip()
    if not email:
        raise ValueError("missing email")
    region = str(record.get("region", "")).strip()
    if not region:
        raise ValueError("missing region")
    return {
        "event": "user.updated",
        "user_id": user_id,
        "email": email,
        "region": region,
    }


def build_account_opened(record: Mapping[str, Any]) -> dict[str, str]:
    # Account event consumed by the ledger projection.
    account_id = str(record.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("missing account_id")
    owner_id = str(record.get("owner_id", "")).strip()
    if not owner_id:
        raise ValueError("missing owner_id")
    currency = str(record.get("currency", "")).strip()
    if not currency:
        raise ValueError("missing currency")
    return {
        "event": "account.opened",
        "account_id": account_id,
        "owner_id": owner_id,
        "currency": currency,
    }


def build_account_closed(record: Mapping[str, Any]) -> dict[str, str]:
    # Account event consumed by the ledger projection.
    account_id = str(record.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("missing account_id")
    owner_id = str(record.get("owner_id", "")).strip()
    if not owner_id:
        raise ValueError("missing owner_id")
    reason = str(record.get("reason", "")).strip()
    if not reason:
        raise ValueError("missing reason")
    return {
        "event": "account.closed",
        "account_id": account_id,
        "owner_id": owner_id,
        "reason": reason,
    }


def build_payment_authorized(record: Mapping[str, Any]) -> dict[str, str]:
    # Payment event consumed by fraud detection and settlement.
    payment_id = str(record.get("payment_id", "")).strip()
    if not payment_id:
        raise ValueError("missing payment_id")
    account_id = str(record.get("account_id", "")).strip()
    if not account_id:
        raise ValueError("missing account_id")
    amount = str(record.get("amount", "")).strip()
    if not amount:
        raise ValueError("missing amount")
    currency = str(record.get("currency", "")).strip()
    if not currency:
        raise ValueError("missing currency")
    return {
        "event": "payment.authorized",
        "payment_id": payment_id,
        "account_id": account_id,
        "amount": amount,
        "currency": currency,
    }


'''.strip()


CREATIVE_WRITING_PROMPT = r'''
Write an original literary science-fiction scene of at least 1,200 words. The scene takes place
in an abandoned greenhouse on the Moon while a station mechanic and an archivist search for the
source of a repeating distress signal. Begin with the mechanic discovering fresh condensation on
a window even though the greenhouse has been without power for twenty years. Let the characters
disagree about whether to restore the local system, gradually reveal a personal reason the
archivist recognizes the signal, and make the restoration create an unexpected but nonviolent
danger.

Use the following background as constraints, not as text to quote. The greenhouse belonged to a
failed agricultural research wing called Selene Nine. Its long central aisle is bordered by dry
hydroponic beds, collapsed irrigation tubes, and cabinets containing seed envelopes whose labels
have faded. A transparent roof looks toward Earth, but electrostatic dust has left only irregular
patches of blue reflected light. Emergency doors divide the greenhouse into three pressure zones.
The first zone is open to the station corridor, the second retains a thin stale atmosphere, and
the third has not been opened since the wing was abandoned. An old maintenance terminal near the
second door controls power, ventilation, irrigation, and the door interlocks. It can restore only
one subsystem at a time until its batteries recharge.

The mechanic, Mara Venn, is practical, tired, and accustomed to repairing systems with incomplete
records. She distrusts unexplained signals because a false evacuation alarm once injured members
of her crew. The archivist, Ilyan Sato, is careful with objects but evasive about his own history.
Years earlier his older sister recorded navigation beacons for remote lunar shelters. The
repeating signal contains a cadence he associates with her, although he initially describes it
only as an obsolete checksum. Neither character knows whether the resemblance is intentional,
accidental, or the result of damaged equipment. Do not confirm a supernatural explanation.

Build the scene through physical investigation rather than exposition. Allow Mara and Ilyan to
notice different details and draw conflicting conclusions. Their dialogue should include pauses,
unfinished thoughts, and practical discussion of pressure seals, battery charge, and signal
routing, but it should remain understandable without technical background. Reveal Ilyan's
connection to the signal gradually through his choices and reactions before he explains it. Give
Mara a credible reason to activate the terminal despite her distrust. When power returns, make
the greenhouse respond through several linked changes rather than a single dramatic switch.

Let the pacing alternate between quiet observation and short bursts of purposeful movement.
Objects should carry evidence of ordinary people who once worked there, but invent those traces
freely rather than turning them into a historical explanation. The characters may disagree about
what a damaged log entry, tool, photograph, or handwritten label means. Give each character at
least one moment in which an assumption proves incomplete. Their relationship may soften under
pressure, but avoid romance, speeches about trust, or a sudden confession that settles their
disagreement. Leave room for ambiguity in both their motives and the greenhouse's behavior.

The resulting danger must be immediate and unexpected but nonviolent: for example, shifting
pressure, rapidly spreading condensation, awakened machinery, obscured visibility, or doors
following an old quarantine routine. It must arise logically from the neglected greenhouse and
force the characters to cooperate. Do not introduce an attacker, alien creature, weapon, rescue
team, or sudden external disaster. Do not resolve whether the distress signal was deliberately
sent. End this excerpt while the characters are still confronting the consequences of their
decision, not with a summary or a neat explanation.

Use vivid sensory details and a restrained emotional tone. Recur to dust, reflected Earthlight,
and dormant seeds, changing what those images mean as the scene develops. Keep the viewpoint
close to Mara without using first person. Avoid headings, an outline, epigraphs, flashback
sections, or authorial commentary. Do not mention these instructions. Continue the scene well
beyond 512 model tokens rather than ending early.
'''.strip()
