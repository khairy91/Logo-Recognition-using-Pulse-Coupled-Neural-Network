import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collections;
import java.util.Vector;

import javax.imageio.ImageIO;

public class rImage {
	int vert, horz;
	Vector<Double> data , v;

	rImage() {
		data = new Vector<Double>();
		v = new Vector<Double>();
		vert = horz = 0;
	}

	rImage(rImage r) {
		data = new Vector<Double>();
		v = new Vector<Double>();
		horz = r.horz;
		vert = r.vert;
		for (int i = 0; i < horz * vert; ++i) {
			data.add(r.data.get(i));
		}
	}

	rImage(int v, int h) {
		data = new Vector<Double>();
		data.setSize(v * h);
		for (int i = 0; i < h * v; ++i)
			data.set(i, 0.0);
		vert = v;
		horz = h;
	}

	void divEqual(double a) {
		for (int i = 0; i < vert * horz; ++i)
			data.setElementAt(data.get(i) / a, i);
	}

	double sum() {
		double res = 0;
		for (int i = 0; i < vert * horz; ++i)
			res += data.get(i);
		return res;
	}

	void loadByte(String path) throws IOException {
		BufferedImage image = ImageIO.read(new File(path));
		int height = image.getHeight(), width = image.getWidth(), rgb, rd, gr, bl, grey;
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				rgb = image.getRGB(j, i);
				rd = (int) ((rgb >> 16) & 0xff);
				gr = (rgb >> 8) & 0xff;
				bl = (rgb >> 0) & 0xff;
				grey = (int) (0.21 * rd + 0.71 * gr + 0.07 * bl);
				data.add((double) grey);
			}
		}
		vert = height;
		horz = width;
	}

	rImage convolve(rImage kern) {
		int kvert = kern.vert, khorz = kern.horz;
		int dv = kvert / 2, dh = khorz / 2;
		rImage ans = new rImage(vert, horz);
		for (int i = 0; i < vert; ++i) {
			for (int j = 0; j < horz; ++j) {
				double sum = 0;
				int ik0 = i - dv, jl0 = j - dh;
				int kx0 = Math.max(0, dh - i), kx1 = Math.min(khorz, horz + dh
						- i);
				int ky0 = Math.max(0, dv - j), ky1 = Math.min(kvert, vert + dv
						- j);
				for (int kx = kx0; kx < kx1; ++kx) {
					int ik = ik0 + kx;
					for (int ky = ky0; ky < ky1; ++ky) {
						int jl = jl0 + ky;
						sum += data.get(ik * horz + jl)
								* kern.data.get(kx * kern.horz + ky);
					}
				}
				ans.data.setElementAt(sum, i * horz + j);
			}
		}
		return ans;
	}

	rImage add(double a) {
		rImage res = new rImage(vert, horz);
		for (int i = 0; i < vert * horz; ++i)
			res.data.setElementAt(data.get(i) + a, i);
		return res;
	}

	rImage mult(double a) {
		rImage res = new rImage(vert, horz);
		for (int i = 0; i < vert * horz; ++i)
			res.data.setElementAt(data.get(i) * a, i);
		return res;
	}

	rImage addMat(rImage a) {
		rImage res = new rImage(vert, horz);
		for (int i = 0; i < vert * horz; ++i)
			res.data.setElementAt(a.data.get(i) + data.get(i), i);
		return res;
	}

	rImage multMat(rImage a) {
		rImage res = new rImage(vert, horz);
		for (int i = 0; i < vert * horz; ++i)
			res.data.setElementAt(a.data.get(i) * data.get(i), i);
		return res;
	}

	int cntWhite() {
		int res = 0;
		double mx = Collections.max(data), mn = Collections.min(data);
		double gain, offset = mn;
		gain = 255.0 / (mx - mn);
		v = (Vector<Double>) data.clone();
		for (int i = 0; i < horz * vert; i++) {
			double x = v.get(i);
			v.setElementAt((x - offset) * gain, i);
		}
		for (int i = 0; i < horz; ++i) {
			for (int j = 0; j < vert; ++j) {
				if (v.get(i * horz + j) == 0)
					res++;
			}
		}
		return res;
	}
	
	

	int saveTarga(String s) throws IOException {
		int res = 0;
		res = cntWhite();
		FileWriter f1 = new FileWriter(s, false);
		f1.write("P1\n# yout01.pbm\n" + Integer.toString(horz) + " "
				+ Integer.toString(vert) + "\n");
		for (int i = 0; i < horz; ++i) {
			for (int j = 0; j < vert; ++j) {
				f1.write((v.get(i * horz + j) > 0 ? "1" : "0") + " ");
				// if (v.get(i * horz + j) == 0)
				// res++;
			}
		}
		f1.flush();
		f1.close();
		return res;
	}
}
