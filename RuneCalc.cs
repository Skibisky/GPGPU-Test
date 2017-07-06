using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Threading.Tasks;
namespace Cudafy_Test
{
	public struct RunData
	{
		public double dataTime;
		public double cpuTime;
		public double gpuTime;
		public double gpuOn;
		public double gpuOff;
		public double sumSq;
	}

	public class RuneCalc
	{
		private static CudafyModule km = null;
		private static GPGPU gpu = null;

		public static int[] stat = null;
		public static int[,] mult = null;
		public static int[,] flat = null;

		private static int[] dev_stat = null;
		private static int[,] dev_mult = null;
		private static int[,] dev_flat = null;

		public static int blockSize = 512;

		[Cudafy]
		public static void calc_r(GThread thread, int n, int[,] build, int[] stat, int[,] mult, int[,] flat, int[,] res)
		{
			int k, id;
			int i = thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x;
			while (i < n)
			{
				for (k = 0; k < 6; k++)
				{
					id = build[i, k];
					for (int j = 0; j < 8; j++)
					{
						res[i, j] = stat[j] * mult[id, j] + flat[id, j];
					}
				}
				i += (thread.blockDim.x * thread.gridDim.x);
			}
		}

		public static void init(eArchitecture archi = eArchitecture.sm_20, bool hasSdk = false, bool generate = false)
		{
			if (archi == eArchitecture.Emulator)
				CudafyModes.Target = eGPUType.Emulator;
			else if (archi >= eArchitecture.OpenCL)
				CudafyModes.Target = eGPUType.OpenCL;
			if (hasSdk)
			{
				// Build the module
				if (generate || CudafyModes.Target != eGPUType.Cuda)
				{
					if (CudafyModes.Target == eGPUType.Cuda)
						CudafyTranslator.Language = eLanguage.OpenCL;
					km = CudafyTranslator.Cudafy(archi);
					km.Serialize("bespoke_" + archi);
				}
				else
				{
					km = new CudafyModule();
					km.SourceCode = System.IO.File.ReadAllText("cuda.cu");
					km.Compile(eGPUCompiler.CudaNvcc);
				}
			}
			else
			{
				// Load the module
				km = CudafyModule.Deserialize(archi.ToString());
			}
			// pretend it has the function it actually has
			if (!generate && !km.Functions.ContainsKey("calc_r"))
				km.Functions.Add("calc_r", new KernelMethodInfo(typeof(RuneCalc), typeof(RuneCalc).GetMethod("calc_r"), eKernelMethodType.Global, false, eCudafyDummyBehaviour.Default, km));
			gpu = CudafyHost.GetDevice(CudafyModes.Target, 0);
			gpu.LoadModule(km);
		}

		public static void Seed()
		{
			if (km == null || gpu == null)
				throw new Exception("please init");

			if (stat == null || mult == null || flat == null)
				throw new Exception("please set data");
			// insert runes here

			dev_stat = gpu.Allocate(stat);
			dev_mult = gpu.Allocate(mult);
			dev_flat = gpu.Allocate(flat);

			gpu.CopyToDevice(stat, dev_stat);
			gpu.CopyToDevice(mult, dev_mult);
			gpu.CopyToDevice(flat, dev_flat);
		}

		public static RunData Execute(int[,] build)
		{
			if (dev_stat == null || dev_mult == null || dev_flat == null)
				throw new Exception("please seed");
			RunData rd = new RunData();

			int[,] dev_build = null;
			int[,] res_g = new int[build.GetLength(0), 8];
			int[,] dev_res = null;
			Console.Write("GPU On         \r");
			rd.gpuOn = MeasureTime(() =>
			{
				dev_build = gpu.Allocate(build);
				dev_res = gpu.Allocate(res_g);
				gpu.CopyToDevice(build, dev_build);
			});

			Console.Write("GPU Run         \r");
			rd.gpuTime = MeasureTime(() =>
			{
				gpu.Launch((build.GetLength(0)) / blockSize, blockSize, "calc_r", (build.GetLength(0)), dev_build, dev_stat, dev_mult, dev_flat, dev_res);
				gpu.Synchronize();
			});

			Console.Write("GPU Off         \r");
			rd.gpuOff = MeasureTime(() =>
			{
				gpu.CopyFromDevice(dev_res, res_g);
				gpu.Free(dev_res);
				gpu.Free(dev_build);
			});

			int[,] res = new int[build.GetLength(0), 8];

			Console.Write("CPU Run         \r");
			rd.cpuTime = MeasureTime(() =>
			{
				Parallel.For(0, build.GetLength(0), (i) =>
				//for (int i = 0; i < build.GetLength(0); i++)
				{
					int k, id;
					for (k = 0; k < 6; k++)
					{
						id = build[i, k];
						for (int j = 0; j < 8; j++)
						{
							res[i, j] = stat[j] * mult[id, j] + flat[id, j];
						}
					}
				});
			});


			Console.CursorLeft = 0; Console.Write("Calc SSE        ");
			for (int i = 0; i < build.GetLength(0); i++)
			{
				for (int j = 0; j < 8; j++)
				{
					var diff = res[i, j] - res_g[i, j];
					rd.sumSq += diff * diff;
				}
			}
			rd.sumSq /= (build.GetLength(0) / 8);

			return rd;
		}

		static double MeasureTime(Action action)
		{
			Stopwatch watch = new Stopwatch();

			watch.Start();
			action.Invoke();
			watch.Stop();

			return watch.ElapsedTicks / (double)Stopwatch.Frequency;
		}
	}
}
