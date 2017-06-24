#pragma once

#include <caffe/caffe.hpp>
#include <caffe/data_transformer.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

#define	AGE_GROUP_NUM	8

using namespace std;
using namespace cv;
using namespace caffe;

typedef double Dtype;

/*
Class : AgeNet

This class is resopnsible for classifying age with given face image.
*/

class AgeNet
{
private:
	string	model_file;			// model file path (deep_age2.prototxt)
	string	weight_file;			// weight file path (age_net.caffemodel)
	string	mean_file;			// mean file path (mean.binaryproto)

	std::shared_ptr<Net<Dtype>> age_net;		// Deep Convolution Network

public:

	// Constructor
	AgeNet(const string _model_file, const string _weight_file, const string _mean_file)
	{
		model_file.assign(_model_file);
		weight_file.assign(_weight_file);
		mean_file.assign(_mean_file);
	}

	// Initialize age_net
	void initNetwork();

	// Get cv::Mat from mean_file
	void getMeanImgFromMeanFile(Mat& _mean_img);

	// Get blob vector which contains 5 input blobs (Details in implementation)
	void makeBlobVecWithCroppedImg(Mat _img, vector<Blob<Dtype> *>& _blob_vec);

	// Classify age and get probability
	int classify(Mat _img, vector<Dtype>& prob_vec);
};
