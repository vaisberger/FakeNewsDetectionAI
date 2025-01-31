from flask import Flask, render_template, request
from app_script import check_news
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        user_text = request.form.get("text")
        result = check_news(user_text)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

