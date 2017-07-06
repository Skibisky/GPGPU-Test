using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Host;

namespace Cudafy_Test
{
	class Program
	{
		static void Main(string[] args)
		{
			try
			{
				if (args == null || args.Length == 0)
				{
					Console.WriteLine("--list for architecture targets");
					Console.WriteLine("--print for some SDK/GPU info");
					Console.WriteLine("--all to iterate over all target");

					return;
				}

				if (args?.FirstOrDefault() == "--print")
				{
					NvccCompilerOptions nvcc;
					if (IntPtr.Size == 8)
						nvcc = NvccCompilerOptions.Createx64();
					else
						nvcc = NvccCompilerOptions.Createx86();

					Console.WriteLine(string.Format("Platform={0}", nvcc.Platform));
					Console.WriteLine("CUDA SDK at " + nvcc.CompilerPath);
					Console.WriteLine("Test: " + nvcc.TryTest());
					Console.WriteLine("Press anykey for cards...");
					Console.ReadKey();
					Console.WriteLine("Reading...");

					foreach (var t in new eGPUType[] { eGPUType.Cuda, eGPUType.Emulator, eGPUType.OpenCL })
					{
						CudafyModes.Target = t;
						printInfo();
					}
					return;
				}

				var ars = new eArchitecture[] {
					eArchitecture.OpenCL,
					eArchitecture.Emulator,
					eArchitecture.sm_10,
					eArchitecture.sm_11,
					eArchitecture.sm_12,
					eArchitecture.sm_13,
					eArchitecture.sm_20,
					eArchitecture.sm_21,
					eArchitecture.sm_30,
					eArchitecture.sm_35,
					eArchitecture.sm_37,
					eArchitecture.sm_50,
					eArchitecture.sm_52,
					eArchitecture.OpenCL11,
					eArchitecture.OpenCL12
				};

				if (args?.FirstOrDefault() == "--list")
				{
					Console.WriteLine(string.Join(Environment.NewLine, ars.Select(a => a + ": " + (int)a)));
					return;
				}

				if (args?.FirstOrDefault() != "--all")
				{
					ars = args.Select(ii => (eArchitecture)int.Parse(ii)).ToArray();
				}

				foreach (var a in ars)
				{
					try
					{
						Random rand = new Random(4);

						Console.WriteLine("Benching " + a);
						Console.WriteLine("Init: " + MeasureTime(() => RuneCalc.init(a, true, true)));

						int[] num_runes = new int[] { 79, 103, 81, 93, 88, 90 };

						int num_stats = 8;

						RuneCalc.flat = new int[num_runes.Sum(), 8];
						RuneCalc.mult = new int[num_runes.Sum(), 8];

						Console.WriteLine("gen: " + MeasureTime(() =>
						{
							for (int slot = 0; slot < 6; slot++)
							{
								for (int rune = 0; rune < num_runes[slot]; rune++)
								{
									for (int s = 0; s < num_stats; s++)
									{
										if (s < 3)
										{
											RuneCalc.flat[rune, s] = rand.Next(0, 50);
											RuneCalc.mult[rune, s] = rand.Next(0, 20);
										}
										else
										{
											RuneCalc.flat[rune, s] = rand.Next(0, 10);
											RuneCalc.mult[rune, s] = 0;
										}
									}
								}
							}

							RuneCalc.stat = new int[] { 3500, 530, 403, 101, 15, 50, 15, 0 };
						}));

						Console.WriteLine("seed: " + MeasureTime(() => RuneCalc.Seed()));

						List<RunData> rd = new List<RunData>();
						int reps = 10;
						int num_builds = 2 << 23; // 2^31 / 32 (int) / 8 (2d size)
						for (int i = 0; i < reps; i++)
						{
							Console.WriteLine("\rRun " + i);
							Console.Write("Building data\r");
							int[,] b = new int[num_builds, 6];
							var d = MeasureTime(() =>
							{
								for (var j = 0; j < num_builds; j++)
								{
									int rn = 0;
									for (int k = 0; k < 6; k++)
									{
										b[j, k] = rand.Next(rn, rn + num_runes[k]);
										rn += num_runes[k];
									}
								}
							});
							var runData = RuneCalc.Execute(b);
							runData.dataTime = d;
							rd.Add(runData);
							Console.CursorTop -= 1;
						}
						Console.WriteLine();
						Console.WriteLine("Av Dat: " + rd.Average(qr => qr.dataTime));
						Console.WriteLine("Av CPU: " + rd.Average(qr => qr.cpuTime));
						Console.WriteLine("Av GPU: " + rd.Average(qr => qr.gpuTime));
						Console.WriteLine("Av On: " + rd.Average(qr => qr.gpuOn));
						Console.WriteLine("Av Off: " + rd.Average(qr => qr.gpuOff));
						Console.WriteLine("Av SPD: " + 100 * (1 / (rd.Average(qr => qr.gpuTime) / rd.Average(qr => qr.cpuTime))));
						Console.WriteLine("Av Ttl: " + 100 * (1 / ((rd.Average(qr => qr.gpuTime) + rd.Average(qr => qr.gpuOn) + rd.Average(qr => qr.gpuOff)) / rd.Average(qr => qr.cpuTime))));
						Console.WriteLine("Av sse: " + rd.Average(qr => qr.sumSq));
						Console.WriteLine();
					}
					catch (Exception e)
					{
						Console.WriteLine(e.GetType() + ": " + e.Message + Environment.NewLine + e.StackTrace);
					}
				}
			}
			finally
			{
				if (System.Diagnostics.Debugger.IsAttached)
				{
					Console.WriteLine("Press anykey to exit...");
					Console.Read();
				}
			}
		}

		static double MeasureTime(Action action)
		{
			Stopwatch watch = new Stopwatch();

			watch.Start();
			action.Invoke();
			watch.Stop();

			return watch.ElapsedTicks / (double)Stopwatch.Frequency;
		}

		static void printInfo()
		{
			int ij = 0;

			foreach (GPGPUProperties prop in CudafyHost.GetDeviceProperties(CudafyModes.Target))
			{
				Console.WriteLine("   --- General Information for device {0} ---", ij);
				Console.WriteLine("Name:  {0}", prop.Name);
				Console.WriteLine("Platform Name:  {0}", prop.PlatformName);
				Console.WriteLine("Device Id:  {0}", prop.DeviceId);
				Console.WriteLine("Compute capability:  {0}.{1}", prop.Capability.Major, prop.Capability.Minor);
				Console.WriteLine("Clock rate: {0}", prop.ClockRate);
				Console.WriteLine("Simulated: {0}", prop.IsSimulated);
				Console.WriteLine();

				Console.WriteLine("   --- Memory Information for device {0} ---", ij);
				Console.WriteLine("Total global mem:  {0}", prop.TotalMemory);
				Console.WriteLine("Total constant Mem:  {0}", prop.TotalConstantMemory);
				Console.WriteLine("Max mem pitch:  {0}", prop.MemoryPitch);
				Console.WriteLine("Texture Alignment:  {0}", prop.TextureAlignment);
				Console.WriteLine();

				Console.WriteLine("   --- MP Information for device {0} ---", ij);
				Console.WriteLine("Shared mem per mp: {0}", prop.SharedMemoryPerBlock);
				Console.WriteLine("Registers per mp:  {0}", prop.RegistersPerBlock);
				Console.WriteLine("Threads in warp:  {0}", prop.WarpSize);
				Console.WriteLine("Max threads per block:  {0}", prop.MaxThreadsPerBlock);
				Console.WriteLine("Max thread dimensions:  ({0}, {1}, {2})", prop.MaxThreadsSize.x,
								  prop.MaxThreadsSize.y, prop.MaxThreadsSize.z);
				Console.WriteLine("Max grid dimensions:  ({0}, {1}, {2})", prop.MaxGridSize.x, prop.MaxGridSize.y,
								  prop.MaxGridSize.z);

				Console.WriteLine();

				ij++;
			}
		}
	}
}
