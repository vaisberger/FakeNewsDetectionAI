import pandas as pd

# שלב 1: טעינת הקובץ
file_path = 'C:\Users\User\PycharmProjects\FakeNewsDetectionAI\data\Cleaned_Dataset_news.csv'
dataset = pd.read_csv(file_path, encoding='latin1')

# שלב 2: ערבוב הנתונים
dataset_shuffled = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# שלב 3: חישוב גדלי החלוקה
train_size = int(0.7 * len(dataset_shuffled))
test_size = int(0.3 * len(dataset_shuffled))

# חלוקה לסטים
train_set = dataset_shuffled[train_size]
test_set = dataset_shuffled[train_size + test_size]

# שלב 4: שמירת הסטים כקבצי CSV
train_set.to_csv('C:/Users/User/PycharmProjects/FakeNewsDetectionAI/data/train_set.csv',index=False, encoding='utf-8')
test_set.to_csv('C:/Users/User/PycharmProjects/FakeNewsDetectionAI/data/test_set.csv',index=False, encoding='utf-8')