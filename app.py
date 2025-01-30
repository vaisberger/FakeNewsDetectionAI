from flask import Flask, render_template, request

app = Flask(__name__)


def check(text):
    return "false"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_text = request.form.get("text")
        result = check(user_text)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)