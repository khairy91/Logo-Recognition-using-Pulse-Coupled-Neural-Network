import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream.GetField;
import java.util.Collections;
import java.util.Scanner;
import java.util.Vector;

import javax.imageio.ImageIO;

public class Main {
	static String[] pics = { "mc1.jpg", "mc2.jpg", "mc3.jpg", "mc4.jpg",
			"pk1.jpg", "pk2.jpg", "pk2.jpg", "prego1.jpg", "hardees1.jpg",
			"londonburger1.jpg", "londonburger2.jpg", "londonburger3.jpg" };
	static String[] names = { "McDonald's", "McDonald's", "McDonald's",
			"McDonald's", "Pizza King", "Pizza King", "Pizza King", "Prego",
			"Hardee's", "London Burger", "London Burger", "London Burger" };
	static int[] rank = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 21, 22, 23, 24 };
	static int classes = pics.length, n = 14, pixels = 90000;
	static Vector<Vector<Double>> FFNNinput;
	static Vector<Vector<Double>> FFNNoutput;
	static backProbagation bb;
	static Vector<Double> mean, stdDev, sum;

	public static void main(String[] args) throws IOException {
		// System.out.println(mainTestingFunction("testPK1.jpg"));
		// System.out.println(mainTestingFunction("testPK2.jpg"));
		// System.out.println(mainTestingFunction("testLB1.jpg"));
		// System.out.println(mainTestingFunction("testLB2.jpg"));
		// System.out.println(mainTestingFunction("testMC2.jpg"));
		Integer cnt = 10;
		bb = new backProbagation(n, 50, classes);
		FFNNinput = new Vector<Vector<Double>>();
		FFNNoutput = new Vector<Vector<Double>>();
		for (int p = 0; p < classes; p++) {
			Vector<Double> out = new Vector<Double>();
			for (int i = 0; i < classes; ++i)
				out.add(0.0);
			out.set(rank[p], 1.0);
			FFNNoutput.add(out);
			BufferedImage image = ImageIO.read(new File(pics[p]));
			int rh = 300, rw = 300;
			BufferedImage r = new BufferedImage(rh, rw, 1);
			Graphics2D g = r.createGraphics();
			g.drawImage(image, 0, 0, rw, rh, null);
			g.dispose();
			image = r;
			ImageIO.write(image, "jpg", new File("resized.jpg"));
			int height = image.getHeight(), width = image.getWidth();
			rImage stim = new rImage();
			stim.loadByte("resized.jpg");
			PCNN net = new PCNN(height, width);
			net.vf = 0;
			stim.divEqual(256);
			int nIterations = n;
			System.out.println("iter = " + p);
			Vector<Double> FFNNin = new Vector<Double>();
			while (nIterations != 0) {
				int x = net.iterate(stim);
				double mx = Collections.max(net.Y.data);
				double mn = Collections.min(net.Y.data);
				int cn = net.Y.cntWhite();
				if (mx > mn && cn > 5000 && cn < (pixels - 5000)) {
					int x1 = net.Y.saveTarga(names[rank[p]] + cnt.toString()
							+ ".pbm");
					FFNNin.add((double) (pixels - x1) / 90000);// scale sa7
					// FFNNin.add((double) (pixels - x1) );// scale sa7
					nIterations--;
					cnt++;
				}
			}
			System.out.println(cnt);
			FFNNinput.add(FFNNin);
		}
		do {
			bb.totError = 0.0;
			for (int i = 0; i < FFNNinput.size(); i++)
				bb.iterate(FFNNinput.get(i), FFNNoutput.get(i));
			System.out.println("TOTERROR == " + bb.totError);
		} while (bb.totError > 2.65);
		// -----------------------------Error----------------------------------
		System.out.println(test("testPK1.jpg"));
		System.out.println(test("testPK2.jpg"));
		System.out.println(test("testMC2.jpg"));
		System.out.println(test("testMC3.jpg"));
		System.out.println(test("testMC4.jpg"));
		System.out.println(test("testMC5.jpg"));
		System.out.println(test("testLB1.jpg"));
		System.out.println(test("testLB2.jpg"));
		System.out.println(bb.inputNeurons + " " + bb.hiddenNeurons + " "
				+ bb.outputNeurons);
		for (int i = 0; i < bb.inputNeurons; i++) {
			for (int j = 0; j < bb.hiddenNeurons; j++)
				System.out.print(bb.hiddenWeights[i][j] + " ");
			System.out.println();
		}
		for (int i = 0; i < bb.hiddenNeurons; i++) {
			for (int j = 0; j < bb.outputNeurons; j++)
				System.out.print(bb.outputWeights[i][j] + " ");
			System.out.println();
		}
	}

	static String mainTestingFunction(String path) throws IOException {
		// String path = "";
		Scanner scn = new Scanner(new File("weights.in"));
		int hiddenNeurons, inputNeurons, outputNeurons;
		inputNeurons = scn.nextInt();
		hiddenNeurons = scn.nextInt();
		outputNeurons = scn.nextInt();
		bb = new backProbagation(inputNeurons, hiddenNeurons, outputNeurons);
		for (int i = 0; i < bb.inputNeurons; ++i)
			for (int j = 0; j < bb.hiddenNeurons; ++j)
				bb.hiddenWeights[i][j] = scn.nextDouble();
		for (int i = 0; i < hiddenNeurons; ++i)
			for (int j = 0; j < outputNeurons; ++j)
				bb.outputWeights[i][j] = scn.nextDouble();
		bb.Y = new Vector<Double>();
		for (int k = 0; k < bb.O.length; ++k)
			bb.Y.add(0.0);
		return test(path);
	}

	static String test(String path) throws IOException {
		BufferedImage image = ImageIO.read(new File(path));
		int rh = 300, rw = 300;
		BufferedImage r = new BufferedImage(rh, rw, 1);
		Graphics2D g = r.createGraphics();
		g.drawImage(image, 0, 0, rw, rh, null);
		g.dispose();
		image = r;
		ImageIO.write(image, "jpg", new File("resized.jpg"));
		int height = image.getHeight(), width = image.getWidth();
		rImage stim = new rImage();
		stim.loadByte("resized.jpg");
		PCNN net = new PCNN(height, width);
		net.vf = 0;
		stim.divEqual(256);
		Vector<Double> res1 = new Vector<Double>();
		Vector<Double> res2 = new Vector<Double>();
		for (int i = 0; i < classes; i++) {
			res1.add(0.0);
			res2.add(0.0);
		}
		Vector<Double> in1 = new Vector<Double>();
		Vector<Double> in2 = new Vector<Double>();
		int nIterations = n;
		Integer cnt = 0;
		while (nIterations != 0) {
			int x = net.iterate(stim);
			double mx = Collections.max(net.Y.data);
			double mn = Collections.min(net.Y.data);
			cnt++;
			int cn = net.Y.cntWhite();
			if (mx > mn && cn > 5000 && cn < (pixels - 5000)) {
				int x1 = net.Y.saveTarga(path + cnt.toString() + ".pbm");
				in1.add((double) (pixels - x1) / 90000);// scale sa7
				in2.add((double) (x1) / 90000);// scale sa7
				nIterations--;
			}
		}
		bb.input = (Vector<Double>) in1.clone();
		bb.computeNetHidden();
		bb.computeNetOutput();
		for (int i = 0; i < classes; i++)
			res1.set(i, res1.get(i) + bb.O[i]);
		bb.input = (Vector<Double>) in2.clone();
		bb.computeNetHidden();
		bb.computeNetOutput();
		for (int i = 0; i < classes; i++)
			res2.set(i, res2.get(i) + bb.O[i]);

		int idx1 = -1, idx2 = -1;
		double mx1 = -1, mx2 = -1;
		for (int i = 0; i < classes; i++) {
			if (res1.get(i) > mx1) {
				mx1 = res1.get(i);
				idx1 = i;
			}
			if (res2.get(i) > mx2) {
				mx2 = res2.get(i);
				idx2 = i;
			}
		}
		if (mx1 > mx2)
			return names[rank[idx1]];
		else
			return names[rank[idx2]];
	}
}
