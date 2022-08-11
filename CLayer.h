#pragma once
#include "Imagelib.h"
#include "CTensor.h"

#define MEAN_INIT 0
#define LOAD_INIT 1

// Layer는 tensor를 입/출력으로 가지며, 특정 operation을 수행하는 Convolutional Neural Netowork의 기본 연산 단위


class Layer {
protected:
	int fK; // kernel size in K*K kernel
	int fC_in; // number of channels
	int fC_out; //number of filters
	string name;
public:
	Layer(string _name, int _fK, int _fC_in, int _fC_out) : name(_name), fK(_fK), fC_in(_fC_in), fC_out(_fC_out) {}
	virtual ~Layer() {}; //가상소멸자 (참고: https://wonjayk.tistory.com/243)
	virtual Tensor3D* forward(const Tensor3D* input) = 0;
	//	virtual bool backward() = 0;
	virtual void print() const = 0;
	virtual void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const = 0;
};


class Layer_ReLU : public Layer {
public:
	Layer_ReLU(string _name, int _fK, int _fC_in, int _fC_out) : Layer::Layer(_name, _fK, _fC_in, _fC_out)
	{
		
		// 동작1: Base class의 생성자를 호출하여 맴버 변수를 초기화(initialization list를 사용)	
	}
	~Layer_ReLU() {}
	Tensor3D* forward(const Tensor3D* input) override {
		
		// 동작1: input tensor에 대해 각 element x가 양수이면 그대로 전달, 음수이면 0으로 output tensor에 전달   
		// 동작2: 이때, output tensor는 동적할당하여 주소값을 반환
		
		int nH, nW, nC;

		input->get_info(nH, nW, nC);

		Tensor3D* output;
		output = new Tensor3D(nH, nW, fC_out);

		int offset = (fK - 1) / 2;

		for (int c = 0; c < fC_in; c++) {
			for (int h = offset; h < nH - offset; h++) {
				for (int w = offset; w < nW - offset; w++) {
					if (input->get_elem(h, w, c) < 0) {
						output->set_elem(h, w, c, 0);
					}
					else output->set_elem(h, w, c, input->get_elem(h, w, c));
				}
			}
		}

		cout << name << " is finished" << endl;
		return output;
	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		
		// 동작: Tensor3D의 get_info()와 마찬가지로 맴버 변수들을 pass by reference로 외부에 전달
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;
	}
	void print() const override {
		
		// 동작: Tensor3D의 print()와 마찬가지로 차원의 크기를 화면에 출력
		cout << name << " : " << fK << "*" << fK << "*" << fC_in << "*" << fC_out << endl;
	}
};


/////////////////////////////////////////////컨볼루션@@@@@@@@@@@@@@@@@@@
class Layer_Conv : public Layer {
private:
	string filename_weight;
	string filename_bias;
	double**** weight_tensor; // fK x fK x _fC_in x _fC_out 크기를 가지는 4차원 배열

	double* bias_tensor;     // _fC_out 크기를 가지는 1차원 배열 (bias는 각 filter당 1개 존재) 

public:
	Layer_Conv(string _name, int _fK, int _fC_in, int _fC_out, int init_type, string _filename_weight = "", string _filename_bias = "") :Layer::Layer(_name, _fK, _fC_in, _fC_out) {
		
		// 동작1: initialization list와 base class의 생성자를 이용하여 맴버 변수를 초기화
		// 동작2: filename_weight와 filename_bias는 LOAD_INIT 모드일 경우 해당 파일로부터 가중치/바이어스를 불러옴
		// 동작3: init() 함수는 init_type를 입력으로 받아 가중치를 초기화 함 
		


		filename_weight = _filename_weight;
		filename_bias = _filename_bias;
		weight_tensor = dmatrix4D(_fK, _fK, _fC_in, _fC_out);
		bias_tensor = dmatrix1D(_fC_out);
		init(init_type);


	}
	void init(int init_type) {
		
		// 동작1: init_type (MEAN_INIT 또는 LOAD_INIT)에 따라 가중치를 다른 방식으로 초기화 함
		// 동작2: MEAN_INIT의 경우 필터는 평균값을 산출하는 필터가 됨 (즉, 모든 가중치 값이 필터의 크기(fK*fK*fC_in)의 역수와 같아짐 (이때 bias는 모두 0으로 설정)
		// 동작3: LOAD_INIT의 경우 filename_weight, filename_bias의 이름을 가지는 파일의 값을 읽어 가중치에 저장(초기화) 함  
		
		if (init_type == MEAN_INIT) {

			for (int out = 0; out < fC_out; out++) {
				for (int in = 0; in < fC_in; in++) {
					for (int h = 0; h < fK; h++) {
						for (int w = 0; w < fK; w++) {

							weight_tensor[h][w][in][out] = 1.0 / (static_cast<double>(fK) * fK * fC_in);

						}

					}
				}
				bias_tensor[out] = 0;
			}
		}
		else if (init_type == LOAD_INIT) {
			ifstream w1(filename_weight);
			ifstream b1(filename_bias);
			double wd = 0.0;
			double bd=0.0;

			for (int i = 0; i < fC_out; i++) {
				b1 >> bd;
				bias_tensor[i]=bd;

			}

			for (int out = 0; out < fC_out; out++) {
				for (int in = 0; in < fC_in; in++) {
					for (int w = 0; w < fK; w++) {
						for (int h = 0; h < fK; h++) {
							w1 >> wd;
							weight_tensor[h][w][in][out]=wd;
						
						}
					}
				}
			}
			
			w1.close();
			b1.close();

		}

	}
	~Layer_Conv() override {

		// 동작1: weight_tensor와 bias_tensor를 동적 할당 해제할 것
		// 함수1: free_dmatrix4D(), free_dmatrix1D() 함수를 사용
		free_dmatrix4D(weight_tensor, fK, fK, fC_in, fC_out);
		free_dmatrix1D(bias_tensor, fC_out);
	}
	Tensor3D* forward(const Tensor3D* input) override {
	
		// 동작1: 컨볼루션 (각 위치마다 y = WX + b)를 수행
		// 동작2: output (Tensor3D type)를 먼저 동적 할당하고 연산이 완료된 다음 pointer를 반환 


		int nH, nW, nC;
		int weight = 0;
		int offset = (fK - 1) / 2;
		input->get_info(nH, nW, nC);


		Tensor3D* output;
		output = new Tensor3D(nH, nW, fC_out);


		for (int out = 0; out < fC_out; out++) {
			for (int in = 0; in < nC; in++) {
				for (int h = offset; h < nH - offset; h++) {
					for (int w = offset; w < nW - offset; w++) {

						for (int ph = 0; ph < fK; ph++) {

							for (int pw = 0; pw < fK; pw++) {

								output->set_elem(h, w, out, output->get_elem(h, w, out) + (input->get_elem(ph + h - offset, pw + w - offset, in) * weight_tensor[ph][pw][in][out]));

							}
						}
					}
				}
			}
		}
		for (int out=0; out < fC_out;out++) {
			for (int h = offset; h < nH - offset; h++) {
				for (int w = offset; w < nW - offset; w++)
				{
					output->set_elem(h, w, out, output->get_elem(h, w, out) + bias_tensor[out]);

				}
			}
		}



		cout << name << " is finished" << endl;
		return output;

	};
	void get_info(string& _name, int& _fK, int& _fC_in, int& _fC_out) const override {
		
		// 동작: Layer_ReLU와 동일
		_name = name;
		_fK = fK;
		_fC_in = fC_in;
		_fC_out = fC_out;

	}
	void print() const override {
	
		// 동작: Layer_ReLU와 동일
		cout << name << " : " << fK << "*" << fK << "*" << fC_in << "*" << fC_out << endl;
	}
};


