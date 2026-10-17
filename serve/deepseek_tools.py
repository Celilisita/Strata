"""Select native DeepSeek tool formatting without changing other templates."""
import html
import json


def tool_dialect(source):
    return 'deepseek_dsml' if '｜DSML｜' in source and 'invoke name=' in source else None


def template_tools(tools, dialect):
    if dialect != 'deepseek_dsml' or tools is None:
        return tools
    return [t if t.get('type') == 'function' else {'type': 'function', 'function': t} for t in tools]


def tool_prefix(dialect, name=None):
    if dialect == 'deepseek_dsml':
        opening = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="'
        return opening + (html.escape(name, quote=True) + '">\n' if name is not None else '')
    return '<tool_call>\n<function=' + (name + '>\n' if name is not None else '')


def output_parser(template, base, **kwargs):
    parser = base(**kwargs)
    if getattr(template, 'tool_dialect', None) == 'deepseek_dsml' and kwargs.get('tools'):
        from serve.deepseek_tool_parser import DsmlOutputParser
        return DsmlOutputParser(parser)
    return parser


def template_caps(template):
    from serve.deepseek_tool_parser import DsmlOutputParser, ROOT
    from serve.frontend import OutputParser
    user = {'role': 'user', 'content': 'strata_caps_user'}
    tools = [{'name': 'strata_caps_call_' + str(i), 'description': 'strata_caps_description',
              'parameters': {'type': 'object', 'properties': {'arg': {'type': 'string'}}}} for i in range(2)]
    def render(messages):
        try:
            return template.render(messages, tools=tools)
        except Exception:
            return ''
    def supported(count):
        calls = [{'name': t['name'], 'arguments': {'arg': 'strata_caps_arg_' + str(i)}}
                 for i, t in enumerate(tools[:count])]
        messages = [user, {'role': 'assistant', 'content': '',
                          'tool_calls': [{'function': c} for c in calls]},
                    {'role': 'tool', 'content': 'strata_caps_result'}, user]
        text = render(messages)
        # Skip the instruction example; inspect actual invokes in replayed history.
        at = text.find('<｜Assistant｜>')
        body = text[at + len('<｜Assistant｜>'):].split('<｜end▁of▁sentence｜>', 1)[0]
        body = body.removeprefix('</think>')
        parser = DsmlOutputParser(OutputParser(thinking=False, tools=tools))
        try:
            events = parser.feed(body) + parser.finish('stop')
        except ValueError:
            return False
        actual = [{'name': e.call.name, 'arguments': e.call.arguments} for e in events if e.kind == 'tool_call']
        return actual == calls and 'strata_caps_result' in text
    prompt = render([user])
    return {'supports_tools': all(s in prompt for s in ('strata_caps_call_0', 'strata_caps_description', ROOT)),
            'supports_tool_calls': supported(1), 'supports_parallel_tool_calls': supported(2),
            'supports_system_role': 'strata_caps_system' in render([{'role': 'system', 'content': 'strata_caps_system'}, user]),
            'supports_preserve_reasoning': 'strata_caps_reasoning' in render(
                [user, {'role': 'assistant', 'content': 'answer', 'reasoning_content': 'strata_caps_reasoning'}, user])}
