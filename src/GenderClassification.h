#pragma once

#include <caffe/caffe.hpp>
#include <caffe/data_transformer.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

#define	MALE	0
#define	FEMALE	1

using namespace std;
using namespace caffe;
using namespace cv;

typedef double Dtype;

/*
Class : GenderNet
	
This class is resopnsible for classifying gender with given face image.
*/

class GenderNet
{
private:
	string	model_file;			// model file path (deep_gender2.prototxt)
	string	weight_file;			// weight file path (gender_net.caffemodel)
	string	mean_file;			// mean file path (mean.binaryproto)

	std::shared_ptr<Net<Dtype>> gender_net;		// Deep Convolution Network

public:

	// Constructor
	GenderNet(const string _model_file, const string _weight_file, const string _mean_file)
	{
		model_file.assign(_model_file);
		weight_file.assign(_weight_file);
		mean_file.assign(_mean_file);
	}

	// Initialize gender_net
	void initNetwork();

	// Get cv::Mat from mean_file
	void getMeanImgFromMeanFile(Mat& _mean_img);

	// Get blob vector which contains 5 input blobs (Details below in implementation)
	void makeBlobVecWithCroppedImg(Mat _img, vector<Blob<Dtype> *>& _blob_vec);

	// Classify gender and get probability
	int classify(Mat _img, vector<Dtype>& _prob_vec);
};
