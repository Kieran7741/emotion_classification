import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
import pickle

# pd.set_option('display.max_columns', None)


def prepare_data():
    """
    Prepare training and test data
    """
    sad_data = pd.read_csv('dataset/sad_data.csv')
    sad_training, sad_test = train_test_split(sad_data, random_state=7)
    # Due to the low number of sad images only 150 happy images are picked
    happy_data = pd.read_csv('dataset/happy_data.csv')[:150]
    happy_training, happy_test = train_test_split(happy_data, random_state=7)
    # neutral_data = pd.read_csv('dataset/neutral_data.csv')
    # neutral_training, neutral_test = train_test_split(neutral_data, random_state=7)

    training_data = pd.concat([sad_training, happy_training])  #, neutral_training])
    testing_data = pd.concat([sad_test, happy_test])  #, neutral_test])

    return training_data, testing_data


train_data, test_data = prepare_data()
train_x, train_y = train_data.drop(columns=['emotion']), train_data['emotion']
test_x, test_y = test_data.drop(columns=['emotion']), test_data['emotion']


model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=600, random_state=7)
model.fit(train_x, train_y)

with open('model.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

prediction = model.predict(test_x)

score = accuracy_score(test_y, prediction)
print(f'Prediction score: {score}')
print(test_y.value_counts())
cm = confusion_matrix(test_y, prediction, normalize=None)
plot = ConfusionMatrixDisplay(cm, ['happy', 'sad'])
plot.plot(xticks_rotation=90)
result_string = f'{model}: Score: {round(score, 3)}'
plot.ax_.set_title(result_string)
plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)
plt.show()

