"""
多國語系
"""

_languageyValues = {}

def getString(language: str, key: str) -> str:
    if language == None:
        language = 'en'
    elif language not in _languageyValues:
        result = _loadLanguage(language)
        if not result:
            language = 'en'
    return _languageyValues[language][key]

def _loadLanguage(language: str):
    try:
        _languageyValues[language] = getattr(__import__('flaskr.Values.strings_' + language, fromlist=['values']), 'values')#相當於from 'Values.strings_'+language imoprt values as languageyValues[language]
        return True
    except (
        Exception
    ) as e:
        hexs = ''
        for c in language:
            hexs += hex(ord(c)) + ' '
        print('language: ' + hexs)
        import traceback
        print(traceback.format_exc())
        return False

_loadLanguage('en')

# if __name__ == "__main__":
#     print(getString('zh', '你好'))
#     print(getString('en', '你好'))
#     print(getString(None, '你好'))