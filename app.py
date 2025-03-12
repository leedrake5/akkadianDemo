from flask import Flask, render_template, request
# Import your T5 model and tokenizer (assuming you use Hugging Face transformers)
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Import your normalization functions
from normalization import (
    unicodeToAscii,
    REMOVE_BRACKETS_TRANS,
    remove_brackets,
    normalize_digits,
    normalize_brackets,
    gap_filler,
    fix_cuneiform_gap,
    fix_suprasigillum,
    read_and_process_file,
    convert,
    collapse_spaces,
    remove_control_characters,
    normalize,
    normalizeString_cuneiform,
    normalizeString_cuneiform_transliterate_translate,
    normalizeString_en,
    trim_singles
)



app = Flask(__name__)

# Load your T5 model (replace 'your-t5-model' with your actual model name/path)
model_name = "Thalesian/AKK_60m"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define your available prompt styles
PROMPT_STYLES = {
    "Translate cuneiform": "Translate Akkadian cuneiform to English: ",
    "Translate transliteration": "Translate complex Akkadian transliteration to English: ",
    "Translate uncertain transliteration": "Translate simple Akkadian transliteration to English: ",
    "Translate English to cuneiform": "Translate English to Akkadian cuneiform: ",
    "Translate English to transliteration": "Translate English to complex Akkadian transliteration: ",
    "Transliterate cuneiform": "Transliterate Akkadian cuneiform to complex Latin characters: "
}

@app.route("/", methods=["GET", "POST"])
def index():
    translation = None
    if request.method == "POST":
        # Get the user-supplied cuneiform text and selected prompt key
        cuneiform_text = request.form.get("cuneiform_text", "")
        prompt_key = request.form.get("prompt")
        
        # Choose the correct normalization function based on the prompt key
        if prompt_key == "Translate cuneiform":
            processed_text = normalizeString_cuneiform(
                cuneiform_text, use_prefix=True, task="Translate", language="Akkadian"
            )
        elif prompt_key == "Translate transliteration":
            processed_text = normalizeString_cuneiform_transliterate_translate(
                cuneiform_text, use_prefix=True, task="Translate", type="origional", language="Akkadian"
            )
        elif prompt_key == "Translate uncertain transliteration":
            processed_text = normalizeString_cuneiform_transliterate_translate(
                cuneiform_text, use_prefix=True, task="Translate", type="simple", language="Akkadian"
            )
        elif prompt_key == "Translate English to cuneiform":
            processed_text = normalizeString_en(
                cuneiform_text, use_prefix=True, task="Translate", target="cuneiform", language="Akkadian"
            )
        elif prompt_key == "Translate English to transliteration":
            processed_text = normalizeString_en(
                cuneiform_text, use_prefix=True, task="Translate", target="transliteration", type="origional", language="Akkadian"
            )
        elif prompt_key == "Transliterate cuneiform":
            processed_text = normalizeString_cuneiform(
                cuneiform_text, use_prefix=True, task="Transliterate", type="origional", language="Akkadian"
            )
        else:
            # Fallback: if no valid prompt is selected, just use a stripped version of the text
            processed_text = trim_singles(cuneiform_text)

        # Tokenize the processed text and generate the model output
        input_ids = tokenizer.encode(processed_text, return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=512)
        translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
    return render_template("index.html", translation=translation, prompt_styles=PROMPT_STYLES)

if __name__ == "__main__":
    app.run(debug=True)
