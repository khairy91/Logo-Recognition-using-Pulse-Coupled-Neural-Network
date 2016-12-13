public class PCNN {
	int vert, horz;
	rImage F, L, U, T, Y, K;
	double af, al, at, beta, vf, vl, vt;

	PCNN() {

	}

	PCNN(int h, int v) {
		horz = h;
		vert = v;
		F = new rImage(v, h);
		L = new rImage(v, h);
		U = new rImage(v, h);
		T = new rImage(v, h);
		Y = new rImage(v, h);
		K = new rImage(4, 4);
		af = 0.1;
		al = 0.1;
		at = 0.5;
		beta = 0.3;
		vf = 0.5;
		vl = 0.2;
		vt = 127.5;
		stdK();
	}

	void stdK() {
		int kv = K.vert, kh = K.horz;
		double val;
		for (int i = 0; i < kv; ++i) {
			for (int j = 0; j < kh; ++j) {
				val = Math.hypot(i - kv / 2, j - kh / 2);
				if (val != 0.0)
					val = 1.0 / val;
				else
					val = 1.0;
				K.data.setElementAt(val, i * kh + j);
			}
		}
	}
	
	int iterate (rImage A) {

		rImage work = Y.convolve(K);
		F = F.mult(Math.exp(-1.0 / af));
		F = F.addMat(work.mult(vf));
		F = F.addMat(A);
		L = L.mult(Math.exp(-1.0 / al));
		L = L.addMat(work.mult(vl));
		rImage tmp = new rImage(L);
		tmp = tmp.mult(beta);
		tmp = tmp.add(1.0);
		U = F.multMat(tmp);

		for (int i = 0 ; i < vert * horz; ++i) {
			double x = 0;
			if (U.data.get(i) > T.data.get(i))
				x = 1.0;
			Y.data.setElementAt(x, i);
		}
		T = T.mult(Math.exp(-1.0 / at));
		T = T.addMat(Y.mult(vt));
		int res = (int)Y.sum();
		return res;
	}
}
