"""Optional explicit service output ceiling; absence preserves existing limits."""

def configured_limit(value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError("max_output must be a positive integer or null")
    return value


def resolve_limit(requested, maximum):
    if maximum is None:
        return requested
    if requested is None or requested <= 0:
        return requested
    if requested > maximum:
        raise ValueError(f"requested max tokens ({requested}) exceeds configured max_output ({maximum})")
    return requested
