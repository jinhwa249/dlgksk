from sklearn.datasets import load_digits
digits = load_digits()

print("Image Data Shape", digits.data.shape)

print("Labek Data Shape", digits.target.shape)

import numpy as np
import matplotlib.pyplot as plt

# plt.figure(figaize=(20,4))
# for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
#     plt.subplot(1,5, index+1)
#     plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
#     plt.title('Traunung: %i\n' % label, fontsize=20)
#     plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, random_atate=0)
from sklearn.linear_model import LogisticRegression
LogisticRegr = LogisticRegression()
LogisticRegr.fit(X_train, y_train)

print(LogisticRegr.predict(X_test[0].reshape(1,-1)))
print(LogisticRegr.predict(X_test[0:10]))

predictions = LogisticRegr.predict(X_test)
score = LogisticRegr.score(X_test, y_test)
print(score)

import numpy as np
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
plt.xlabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Acccuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15);
plt.show();
