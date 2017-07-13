from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC,SVC
import numpy as np
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split

class SVM:

	def set_features_labels(self):
		labels = np.load("all_labels.npy")
		features = np.load("all_features.npy")

		return features,labels
		
	def svm(self):
		features,labels = self.set_features_labels()
		scores1 = []
		scores2 = []
		scores3 = []

		for k in range(10):
			indices = np.random.permutation(len(features))
			features = features[indices]
			labels = labels[indices]
			trX,teX,trY,teY = train_test_split(features,labels,test_size = 0.30,random_state = 1)

			clf = OneVsRestClassifier(LinearSVC())
			clf = clf.fit(trX,trY)
			pred = clf.predict(teX)
			score1 = precision_score(teY,pred,average = "micro")
			score2 = recall_score(teY,pred,average = "micro")
			score3 = f1_score(teY,pred,average = "micro")
			scores1.append(score1)
			scores2.append(score2)
			scores3.append(score3)
		
		print "micro precision    : %.2f" % np.array(scores1).mean()
		print "micro recall    : %.2f" % np.array(scores2).mean()
		print "micro F1        : %.2f" % np.array(scores3).mean()

if __name__ == "__main__":
	s = SVM()
	s.svm()