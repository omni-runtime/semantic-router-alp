"""Trusted whole-response requirements, independent of the model wire format."""
from __future__ import annotations

from .catalog import PayloadConstraints, stable_json
from .errors import ALPError


def merge_payload(base, extra):
    for path in base.fixed_values.keys() & extra.fixed_values.keys():
        if stable_json(base.fixed_values[path]) != stable_json(extra.fixed_values[path]):
            raise ValueError("Conflicting host fixed values")
    capabilities = {**base.capabilities}
    if capabilities and not set(extra.capabilities).issubset(capabilities):
        raise ValueError("Member capabilities exceed the host domain")
    for name, contract in extra.capabilities.items():
        before = capabilities[name].model_dump(exclude_none=True) if name in capabilities else {}
        after = contract.model_dump(exclude_none=True)
        if any(stable_json(before[k]) != stable_json(after[k]) for k in before.keys() & after.keys()):
            raise ValueError("Conflicting capability contracts")
        capabilities[name] = {**before, **after}
    return PayloadConstraints(
        required_fields=list(dict.fromkeys([*base.required_fields, *extra.required_fields])),
        forbidden_fields=list(dict.fromkeys([*base.forbidden_fields, *extra.forbidden_fields])),
        fixed_values={**base.fixed_values, **extra.fixed_values}, capabilities=capabilities,
    )


def member_catalog(catalog, index):
    plan = catalog.response_constraints
    if not plan or not plan.members:
        return catalog
    if not 0 <= index < len(plan.members):
        raise ALPError("HOST_RESPONSE_COUNT_MISMATCH", "Unexpected response member.", 502)
    member = plan.members[index]
    result = catalog.model_copy(deep=True)
    result.response_constraints = None
    try:
        result.payload_constraints[member.operation] = merge_payload(
            result.payload_constraints.get(member.operation, PayloadConstraints()), member.payload)
    except ValueError:
        raise ALPError("INVALID_TASK_CONSTRAINT", "Conflicting response member requirements.", 422) from None
    return result


def validate_plan(version, operations, catalog):
    plan = catalog.response_constraints
    if plan is None:
        return
    if version != "0.4.0":
        raise ALPError("INVALID_TASK_CONSTRAINT", "Response constraints require ALP 0.4.", 422)
    for i, member in enumerate(plan.members or []):
        if member.operation not in operations or member.payload.capabilities and member.operation != "agent_definition_generate":
            raise ALPError("INVALID_TASK_CONSTRAINT", "A response member is outside the operation domain.", 422)
        member_catalog(catalog, i)


def validate_response(actions, catalog):
    plan = catalog.response_constraints
    if plan is None:
        return
    if not plan.min_calls <= len(actions) <= plan.max_calls:
        raise ALPError("HOST_RESPONSE_COUNT_MISMATCH", "The response violates the host call count.", 502,
                       [{"path": "/", "expected_min": plan.min_calls, "expected_max": plan.max_calls,
                         "actual_count": len(actions)}])
    for i, member in enumerate(plan.members or []):
        if actions[i]["operation"] != member.operation:
            raise ALPError("HOST_RESPONSE_ORDER_MISMATCH", "The response violates the host member order.", 502,
                           [{"action_index": i, "path": "/operation"}])


def response_coverage(catalog):
    plan = catalog.response_constraints
    return {"count": "bound" if plan else "not_task_bound",
            "min_calls": plan.min_calls if plan else 1,
            "max_calls": plan.max_calls if plan else 16,
            "members": "ordered_and_bound" if plan and plan.members else "not_task_bound"}
