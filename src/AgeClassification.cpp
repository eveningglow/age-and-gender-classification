#include "AgeClassification.h"

/*
Method : initNetwork

Read network architecture from model_file and apply weights from weight_file
*/
void AgeNet::initNetwork()
{
	NetParameter net_param;
	ReadNetParamsFromTextFileOrDie(this->model_file, &net_param);
	
	this->age_net = make_shared<Net<Dtype>>(net_param);
	age_net->CopyTrainedLayersFrom(this->weight_file);
}

/*
Method : getMeanImgFromMeanFile

Get cv::Mat from mean_file.
This work goes through the steps below.

1. Get BlobProto from mean_file.
2. Make Blob using BlobProto
3. Make three cv::Mat(one Mat per channel) using Blob
4. Merge three channels Mat.
5. Get cv::Scalar(one value per channel) with the mean values of each channel(mean value of R channel,... and so on).
6. Make mean img with this scalar.

Param :
1. _mean_img : [output] mean image
*/
void AgeNet::getMeanImgFromMeanFile(Mat& _mean_img)
{
	BlobProto blob_proto;
	ReadProtoFromBinaryFile(this->mean_file.c_str(), &blob_proto);

	Blob<Dtype> mean_blob;
	mean_blob.FromProto(blob_proto);

	vector<Mat> channels;
	Dtype * mean_data = mean_blob.mutable_cpu_data();

	for (int i = 0; i < mean_blob.channels(); i++)
	{
		Mat channel(mean_blob.width(), mean_blob.height(), CV_8UC1, mean_data);
		channels.push_back(channel);
		mean_data += mean_blob.width() * mean_blob.height();
	}

	Mat rgb_mean_img;
	cv::merge(channels, rgb_mean_img);

	Scalar channel_mean = cv::mean(rgb_mean_img);

	_mean_img = cv::Mat(rgb_mean_img.rows, rgb_mean_img.cols, rgb_mean_img.type(), channel_mean);
}

/*
Method : makeBlobVecWithCroppedImg

This is a kind of useful technique to make performance better.

1. Resize image to 256 x 256
2. Subtract mean image.
3. Get 5(left top, right top, left bottom, right bottom, center) cropped images with size of 227 x 227.
4. Make Blob of those 5 images (Each blob is going to input blob of the network).
5. Make Blob vector.

Param :
1. _img : [in] input image
2. _blob_vec : [out] output blob vector made by five cropped image of _img.
*/
void AgeNet::makeBlobVecWithCroppedImg(Mat _img, vector<Blob<Dtype> *>& _blob_vec)
{
	Mat resized_img;
	Mat mean_img;
	Mat normalized_img;

	cv::resize(_img, resized_img, cv::Size(256, 256));

	this->getMeanImgFromMeanFile(mean_img);
	subtract(resized_img, mean_img, normalized_img);

	Mat lt_img = normalized_img(cv::Rect(0, 0, 227, 227));
	Mat rt_img = normalized_img(cv::Rect(29, 0, 227, 227));
	Mat lb_img = normalized_img(cv::Rect(0, 29, 227, 227));
	Mat rb_img = normalized_img(cv::Rect(29, 29, 227, 227));
	Mat ctr_img = normalized_img(cv::Rect(14, 14, 227, 227));

	Blob<Dtype> * lt_blob = new Blob<Dtype>(1, 3, 227, 227);
	Blob<Dtype> * rt_blob = new Blob<Dtype>(1, 3, 227, 227);
	Blob<Dtype> * lb_blob = new Blob<Dtype>(1, 3, 227, 227);
	Blob<Dtype> * rb_blob = new Blob<Dtype>(1, 3, 227, 227);
	Blob<Dtype> * ctr_blob = new Blob<Dtype>(1, 3, 227, 227);

	TransformationParameter trans_param;
	DataTransformer<Dtype> transformer(trans_param, caffe::TEST);

	transformer.Transform(lt_img, lt_blob);
	transformer.Transform(rt_img, rt_blob);
	transformer.Transform(lb_img, lb_blob);
	transformer.Transform(rb_img, rb_blob);
	transformer.Transform(ctr_img, ctr_blob);

	_blob_vec.push_back(lt_blob);
	_blob_vec.push_back(rt_blob);
	_blob_vec.push_back(lb_blob);
	_blob_vec.push_back(rb_blob);
	_blob_vec.push_back(ctr_blob);
}

/*
Method : classify

Classify age of person in the image.

1. Crop image into 5 images and get a blob vector which contains 5 blobs (makeBlobVecWithCroppedImg).
2. Get probabilities of each image. Finally, you got 5 probabilities.
3. Get average probabilities of these 5 blobs.
4. Get the biggest probability.

Param :
1. _img : [in] input image
2. _prob_vec : [out] probabilities of each age
*/
int AgeNet::classify(Mat _img, vector<Dtype>& prob_vec)
{
	vector<Blob<Dtype> *> input_blob_vec;

	for (int i = 0; i < AGE_GROUP_NUM; i++)
		prob_vec.push_back(0);

	this->makeBlobVecWithCroppedImg(_img, input_blob_vec);

	for (auto cropped_blob : input_blob_vec)
	{
		this->age_net->input_blobs()[0]->CopyFrom(*(cropped_blob));
		this->age_net->Forward();
		Dtype * result = this->age_net->output_blobs()[0]->mutable_cpu_data();

		for (int i = 0; i < AGE_GROUP_NUM; i++)
		{
			prob_vec[i] += result[i];
		}
	}

	std::for_each(prob_vec.begin(), prob_vec.end(), [&](Dtype& prob) {prob /= input_blob_vec.size(); });
	auto iter = std::max_element(prob_vec.begin(), prob_vec.end(), [&](Dtype i, Dtype j)->bool {return (i < j); });
	
	return (iter - prob_vec.begin());
}
