#pragma once
#include "Imagelib.h"

// Tensor3D�� ũ�Ⱑ (nH x nW x nC)�� 3���� tensor�� ������

class Tensor3D {
private:
	double*** tensor;
	int nH; // height
	int nW; // width
	int nC; // channel
public:
	Tensor3D(int _nH, int _nW, int _nC) : nH(_nH), nW(_nW), nC(_nC) {
		
		// ����: 1)3���� ����� �����Ҵ��Ͽ�, tensor�� ���� �ּҰ��� ����
		//       2)�� �� ��� element�� ���� 0���� �ʱ�ȭ
		// ����Լ�: dmatrix3D(): 3���� ����� ���� �Ҵ��ؼ� pointer�� ��ȯ�ϴ� �Լ�
		tensor = dmatrix3D(_nH, _nW, _nC);

		for (int c = 0; c < _nC; c++) {
			for (int h = 0; h < _nH; h++) {
				for (int w = 0; w < _nW; w++) {
					tensor[h][w][c] = 0;
				}
			}
		}

	}
	~Tensor3D() {
		
		// ����: 3���� ���� �迭�� tensor�� �Ҵ� ����
		// ����Լ�: free_dmatrix3D(): 3���� ���� �Ҵ�� ����� �Ҵ� �����ϴ� �Լ�
		free_dmatrix3D(tensor, nH, nW, nC);

	}
	void set_elem(int _h, int _w, int _c, double _val) { tensor[_h][_w][_c] = _val; }
	double get_elem(int _h, int _w, int _c)	const {
		
		// ����: ��=_h, ��= _w, ä��= _c ��ġ element�� ��ȯ


		return tensor[_h][_w][_c];
	}

	void get_info(int& _nH, int& _nW, int& _nC) const {
		
		// ����: ����� ����(nH, nW, nC)�� pass by reference�� ��ȯ
		_nH = nH;
		_nW = nW;
		_nC = nC;
	}

	void set_tensor(double*** _tensor) { tensor = _tensor; }
	double*** get_tensor() const { return tensor; }

	void print() const {
	
		// ����: ����� ũ�� (nH*nW*nC)�� ȭ�鿡 ���
		cout << nH << "*" << nW << "*" << nC << endl;
	}
};