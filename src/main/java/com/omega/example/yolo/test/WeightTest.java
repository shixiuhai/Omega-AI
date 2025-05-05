package com.omega.example.yolo.test;

import jcuda.Sizeof;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class WeightTest {
       /**
     * 程序的入口点
     * 该方法尝试打开一个二进制文件，并读取其中的数据
     * 数据包括版本信息（主版本号、次版本号、修订号）和模型的训练信息（已训练的次数）
     * 随后，读取文件中的前10000个浮点数并打印到控制台
     */
    public static void main(String[] args) {
        try (RandomAccessFile file = new RandomAccessFile("H:\\voc\\yolo-weights\\yolov3-tiny.weights", "r")) {
            // 读取文件中的主版本号
            int major = readInt(file);
            // 读取文件中的次版本号
            int minor = readInt(file);
            // 读取文件中的修订号
            int revision = readInt(file);
            // 读取文件中表示已训练次数的大整数
            long seen = readBigInt(file);
            // 打印读取到的版本信息和训练次数
            System.out.println("major:" + major);
            System.out.println("minor:" + minor);
            System.out.println("revision:" + revision);
            System.out.println("seen:" + seen);
            // 循环读取并打印文件中的前10000个浮点数
            for (int i = 0; i < 10000; i++) {
                System.out.println(readFloat(file));
            }
        } catch (Exception e) {
            // 异常处理：打印异常的堆栈跟踪信息
            e.printStackTrace();
        }
    }

    public static long readBigInt(RandomAccessFile inputStream) throws IOException {
        long retVal;
        byte[] buffer = new byte[Sizeof.LONG];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getLong();
        return retVal;
    }

    public static int readInt(RandomAccessFile inputStream) throws IOException {
        int retVal;
        byte[] buffer = new byte[Sizeof.INT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getInt();
        return retVal;
    }

    public static float readFloat(RandomAccessFile inputStream) throws IOException {
        float retVal;
        byte[] buffer = new byte[Sizeof.FLOAT];
        inputStream.readFully(buffer);
        ByteBuffer wrapped = ByteBuffer.wrap(buffer);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        retVal = wrapped.getFloat();
        return retVal;
    }

    public static void readFloat(RandomAccessFile inputStream, float[] data) throws IOException {
        for (int i = 0; i < data.length; i++) {
            data[i] = readFloat(inputStream);
        }
    }
}

