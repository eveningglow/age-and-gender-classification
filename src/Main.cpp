#include "GenderClassification.h"
#include "AgeClassification.h"

using namespace std;
using namespace cv;
using namespace caffe;

/*
Method : printResult

Print classify result on console.

Param :
1. _gender : [in] classified result
2. _age : [in] classified result
3. _prob_gender : [in] probabilities of male and female
4. _prob_age : [in] probabilities of each age
*/
void printResult(int _gender, int _age, vector<Dtype> _prob_gender, vector<Dtype> _prob_age)
{
	string gender;
	string age;

	// multiply 100 of each probability to make it as %
	for_each(_prob_gender.begin(), _prob_gender.end(), [&](Dtype& prob) {prob *= 100; });
	for_each(_prob_age.begin(), _prob_age.end(), [&](Dtype& prob) {prob *= 100; });

	// Convert int _gender into string
	switch (_gender)
	{
	case(0):
		gender = "Male"; break;
	case(1):
		gender = "Female"; break;
	}

	// Convert int _age into string
	switch (_age)
	{
	case(0):
		age = "0 - 2"; break;
	case (1):
		age = "4 - 6"; break;
	case (2):
		age = "8 - 13";	break;
	case (3):
		age = "15 - 20"; break;
	case (4):
		age = "25 - 32"; break;
	case (5):
		age = "38 - 43"; break;
	case (6):
		age = "48 - 53"; break;
	case (7):
		age = "60 -"; break;
	}

	// Print result
	cout << endl;
	cout << " =============== Prediction ===============" << endl << endl;
	cout << "1. Gender" << endl << endl;
	cout << " Male (" << _prob_gender[0] << " %)" << endl;
	cout << " Female (" << _prob_gender[1] << " %)" << endl;
	cout << " => " << gender << endl << endl;

	cout << "2. Age " << endl << endl;
	cout << " 0 - 2 (" << _prob_age[0] << " %)" << endl;
	cout << " 4 - 6 (" << _prob_age[1] << " %)" << endl;
	cout << " 8 - 13 (" << _prob_age[2] << " %)" << endl;
	cout << " 15 - 20 (" << _prob_age[3] << " %)" << endl;
	cout << " 25 - 32 (" << _prob_age[4] << " %)" << endl;
	cout << " 38 - 43 (" << _prob_age[5] << " %)" << endl;
	cout << " 48 - 53 (" << _prob_age[6] << " %)" << endl;
	cout << " 60 - (" << _prob_age[7] << " %)" << endl;
	cout << " => " << age << endl << endl;

	cout << " ==========================================" << endl;
}

int main(int argc, char* argv[])
{
	if (argc != 7)
	{
		cout << "Command shoud be like ..." << endl;
		cout << "AgeAndGenderClassification \"GENDER_NET_MODEL_FILE_PATH\" \"GENDER_NET_WEIGHT_FILE_PATH\" "; 
		cout <<	" \"AGE_NET_MODEL_FILE_PATH\" \"AGE_NET_WEIGHT_FILE_PATH\" \"MEAN_FILE_PATH\" \"TEST_IMAGE\" " << endl;
		
		return 0;
	}

	// Get each file path
	string gender_model(argv[1]);
	string gender_weight(argv[2]);
	string age_model(argv[3]);
	string age_weight(argv[4]);
	string mean_file(argv[5]);
	string test_image(argv[6]);

	// Probability vector
	vector<Dtype> prob_age_vec;
	vector<Dtype> prob_gender_vec;

	// Set mode
	Caffe::set_mode(Caffe::GPU);

	// Make GenderNet and AgeNet
	GenderNet gender_net(gender_model, gender_weight, mean_file);
	AgeNet age_net(age_model, age_weight, mean_file);

	// Initiailize both nets
	gender_net.initNetwork();
	age_net.initNetwork();
	
	// Classify and get probabilities
	Mat test_img = imread(test_image, CV_LOAD_IMAGE_COLOR);
	int gender = gender_net.classify(test_img, prob_gender_vec);
	int age = age_net.classify(test_img, prob_age_vec);

	// Print result and show image
	printResult(gender, age, prob_gender_vec, prob_age_vec);
	imshow("AgeAndGender", test_img);
	waitKey(0);
}
