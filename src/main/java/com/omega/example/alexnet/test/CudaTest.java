package com.omega.example.alexnet.test;
import jcuda.driver.CUdevice;
import jcuda.driver.CUcontext;
import jcuda.driver.JCudaDriver;

public class CudaTest {
    public static void main(String[] args) {
        JCudaDriver.setExceptionsEnabled(true);
        JCudaDriver.cuInit(0);
        CUdevice device = new CUdevice();
        JCudaDriver.cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxCreate(context, 0, device);
        System.out.println("CUDA initialized successfully.");
    }
}