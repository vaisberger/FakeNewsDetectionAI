import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# קריאה של הדאטהסט
file_path = 'Processed_Dataset_news.csv'  # עדכן את הנתיב במידת הצורך
data = pd.read_csv(file_path, low_memory=False)

# הדפסת שמות העמודות והסוגים שלהן
print(data.dtypes)

# סינון העמודות שמכילות נתונים מספריים בלבד (כולל עמודות מעובדות כמו word count, sentence count)
X = data.select_dtypes(include=['number'])  # שימוש בכל העמודות המספריות

# עמודת היעד (label)
y = data['lable']  # עדכן אם השם שונה

# חלוקה לנתוני אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# קנה מידה של התכונות המספריות באמצעות StandardScaler
scaler = StandardScaler()

# החלת סטנדרטיזציה על נתוני האימון והבדיקה
X_train_scaled = scaler.fit_transform(X_train)  # למידת הסטטיסטיקות מהאימון
X_test_scaled = scaler.transform(X_test)  # יישום הטרנספורמציה על הבדיקה

# המרת הנתונים ל-DataFrame עם שמות העמודות המקוריים
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# הוספת עמודת היעד (label) בחזרה
X_train_scaled_df['lable'] = y_train.reset_index(drop=True)
X_test_scaled_df['lable'] = y_test.reset_index(drop=True)

# שמירת הדאטה המעובד בקבצים חדשים
X_train_scaled_df.to_csv('Processed_Train_Dataset_SVM.csv', index=False)
X_test_scaled_df.to_csv('Processed_Test_Dataset_SVM.csv', index=False)

print("Data has been scaled and saved as 'Processed_Train_Dataset_SVM.csv' and 'Processed_Test_Dataset_SVM.csv'.")
