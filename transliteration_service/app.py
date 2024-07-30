from flask import Flask, request, jsonify
from ai4bharat.transliteration import XlitEngine

app = Flask(__name__)

transliterate = XlitEngine("si", beam_width=10, src_script_type="en")
romanizer = XlitEngine(src_script_type="indic", beam_width=10, rescore=False)

@app.route('/transliterate', methods=['POST'])
def transliterate_text():
    data = request.json
    input_text = data.get('input')
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    out = transliterate.translit_sentence(input_text, lang_code="si")
    return jsonify({'transliterated': out})

@app.route('/romanize', methods=['POST'])
def romanize_text():
    data = request.json
    input_text = data.get('input')
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400

    # Romanize the transliterated input text
    test = romanizer.translit_sentence(input_text, lang_code="si")
    return jsonify({'romanized': test})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
