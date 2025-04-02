package com.omega.common.utils;

import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.util.Arrays;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

import com.omega.engine.nn.network.utils.ModelUtils;

import jcuda.Sizeof;

/**
 * 
 */
public class MFCC {
	private final static int NUM_MEL_FILTERS = 26;
	private final static double LOWER_FILTER_FREQUENCY = 80.00;
	private final static double UPPER_FILTER_FREQUENCY = 15000.00;
	private final static int LIFTER = 13;
	private static int sampleRate = 44100;
	private static int frameSize = 2048;

	public static double f2mel(double freq) {
		return 2595 * Math.log10(1 + freq / 700);
	}

	public static int[] fftBinIndices() {
		final int cBin[] = new int[NUM_MEL_FILTERS + 2];
		cBin[0] = (int) Math.round(LOWER_FILTER_FREQUENCY / sampleRate * frameSize);// cBin0
		cBin[cBin.length - 1] = (frameSize / 2);// cBin24
		for (int i = 1; i <= NUM_MEL_FILTERS; i++) {// from cBin1 to cBin23
			final double fc = centerFreq(i);// center freq for i th filter
			cBin[i] = (int) Math.round(fc / sampleRate * frameSize);
		}
		return cBin;
	}

	public static double[] melFilter(double bin[], int cBin[]) {
		final double temp[] = new double[NUM_MEL_FILTERS + 2];
		for (int k = 1; k <= NUM_MEL_FILTERS; k++) {
			double num1 = 0.0, num2 = 0.0;
			for (int i = cBin[k - 1]; i <= cBin[k]; i++) {
				num1 += ((i - cBin[k - 1] + 1) / (cBin[k] - cBin[k - 1] + 1)) * bin[i];
			}

			for (int i = cBin[k] + 1; i <= cBin[k + 1]; i++) {
				num2 += (1 - ((i - cBin[k]) / (cBin[k + 1] - cBin[k] + 1))) * bin[i];
			}

			temp[k] = num1 + num2;
		}
		final double fBank[] = new double[NUM_MEL_FILTERS];
		System.arraycopy(temp, 1, fBank, 0, NUM_MEL_FILTERS);
		return fBank;
	}

	public static double centerFreq(int i) {
		final double melFLow = f2mel(LOWER_FILTER_FREQUENCY);
		final double melFHigh = f2mel(UPPER_FILTER_FREQUENCY);
		final double temp = melFLow + ((melFHigh - melFLow) / (NUM_MEL_FILTERS + 1)) * i;
		return inverseMel(temp);
	}

	public static double inverseMel(double x) {
		final double temp = Math.pow(10, x / 2595) - 1;
		return 700 * (temp);
	}

	public static double[] dct(double x[]) {
		final double cepc[] = new double[x.length];
		// perform DCT
		for (int n = 1; n <= x.length; n++) {
			for (int i = 1; i <= NUM_MEL_FILTERS; i++) {
				cepc[n - 1] += x[i - 1] * Math.cos(Math.PI * (n - 1) / NUM_MEL_FILTERS * (i - 0.5));
			}
		}
		return cepc;
	}

	/**
	 * 获取mfcc特征
	 * @param waveform
	 * @return
	 */
	public static double[] mfcc(double[] waveform) {
		double[] magSpec = null; // getMagSpectrum 这里就是傅里叶要获取的mag，fft获取实部和虚部开根号计算得到

		final int cBin[] = MFCC.fftBinIndices();
		final double fBank[] = MFCC.melFilter(magSpec, cBin);
		final double f[] = Arrays.stream(fBank).map(d -> Math.log(d) < -50 ? -50 : Math.log(d)).toArray();
		double[] mfcc = MFCC.dct(f);
		return Arrays.copyOfRange(mfcc, 0, LIFTER);
	}
	
	/**
	 * 获取fbank特征
	 * @param waveform
	 * @return
	 */
	public static double[] getFbank(double[] waveform) {
		double[] magSpec = null; // getMagSpectrum 这里就是傅里叶要获取的mag，fft获取实部和虚部开根号计算得到

		final int cBin[] = MFCC.fftBinIndices();
		// 这里是获取的fbank非db
		final double fBank[] = MFCC.melFilter(magSpec, cBin);
		// 这是db数据，你可以使用这个f[]作为向量输入
		final double f[] = Arrays.stream(fBank).map(d -> Math.log(d) < -50 ? -50 : Math.log(d)).toArray();

		return f;
	}
	
	public static void main(String[] args) {
        // 文件路径
        String filePath = "H://A0001_S0003_0_G4001_G4002_103.wav";
        File file = new File(filePath);
        
        try {
            // 获取音频输入流
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
            
            // 获取音频格式信息
            AudioFormat format = audioInputStream.getFormat();
            System.out.println("FrameRate:"+format.getFrameRate());
            System.out.println("FrameSize:"+format.getFrameSize());
            System.out.println("Format: " + format);
            
            float pre_emphasis  = 0.97f;
            
            // 读取音频数据（可选，例如转换成字节数组）
            byte[] bytes = new byte[audioInputStream.available()];
            audioInputStream.read(bytes);
            int[] cache = new int[bytes.length/2];
            float[] emphasized_signal = new float[cache.length];
            byte[] buffer = new byte[Sizeof.SHORT];
            for(int i = 0;i<bytes.length/2;i++) {
            	buffer[0] = bytes[i * Sizeof.SHORT];
            	buffer[1] = bytes[i * Sizeof.SHORT + 1];
            	cache[i] = ModelUtils.readShort(buffer);
            }
//            emphasized_signal[0] = cache[0];
//            for(int i = 1;i<cache.length;i++) {
//            	emphasized_signal[i] = cache[i] - pre_emphasis * cache[i - 1];
//            }
            for(int i = 0;i<cache.length;i++) {
            	emphasized_signal[i] = cache[i];
            }
            System.out.println(cache.length);
            System.out.println(JsonUtils.toJson(cache));
            System.out.println(JsonUtils.toJson(emphasized_signal));
            int fr = (int) format.getFrameRate();
            boolean remove_dc_offset = true;
            float[][] strided = get_strided(emphasized_signal, fr, 0.025f, 0.01f, remove_dc_offset, pre_emphasis);
            System.out.println(JsonUtils.toJson(strided));

            // 处理或播放音频数据...
            // 例如，你可以将bytes发送到另一个处理函数或播放设备
            System.out.println("Read " + bytes.length + " bytes of audio data.");
            
        } catch (UnsupportedAudioFileException e) {
            System.err.println("The specified audio file is not supported.");
            e.printStackTrace();
        } catch (IOException e) {
            System.err.println("Error reading the file.");
            e.printStackTrace();
        }
    }
	
	public static float[] getWindowFunction(int window_size) {
		/**
		 * POVEY = HAMMING.pow(0.85)
		 */
		float[] window_function = new float[window_size];
		
		for(int i = 0;i<window_size;i++) {
			window_function[i] = (float) Math.pow((0.5d - 0.5d * Math.cos(2*Math.PI*i/(window_size - 1))), 0.85d); 
		}
		
		return window_function;
	}
	
	public static float[][] get_strided(float[] emphasized_signal,int fs,float flen,float fss,boolean remove_dc_offset,float pre_emphasis) {

		int window_size = Math.round(fs * flen);
		int window_shift = Math.round(fs * fss);
		int num = emphasized_signal.length;
		int m = 1 + (num - window_size) / window_shift;
		/**
		 * padding
		 */
        int padding = new BigInteger("2").pow(new BigInteger((window_size - 1) + "").bitLength()).intValue();
        
		float[][] strided = new float[m][padding];
		float[] mean = new float[m];
		for(int i = 0;i<m;i++) {
			float mv = 0.0f;
			for(int j = 0;j<window_size;j++) {
				float val = emphasized_signal[i * window_shift + j];
				strided[i][j] = val;
				mv += val / window_size;
			}
			mean[i] = mv;
		}
		
		float[] getWindowFunction = getWindowFunction(window_size);
//		System.err.println(JsonUtils.toJson(getWindowFunction));

		float preVal = 0.0f;
		if(remove_dc_offset) {
			for(int i = 0;i<m;i++) {
				for(int j = 0;j<window_size;j++) {
					float tmp = (strided[i][j] - mean[i]);
					if(j == 0) {
						strided[i][j] = (tmp - pre_emphasis * tmp) * getWindowFunction[j];
					}else {
						strided[i][j] = ((strided[i][j] - mean[i]) - pre_emphasis * preVal) * getWindowFunction[j];
					}
					preVal = tmp;
				}
			}
		}

		System.err.println(JsonUtils.toJson(strided[1]));
		return strided;
	}
	
}