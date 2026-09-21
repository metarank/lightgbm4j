package io.github.metarank.lightgbm4j;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_long_long;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.lang.management.ManagementFactory;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;

import static com.microsoft.ml.lightgbm.lightgbmlib.LGBM_BoosterDumpModelSWIG;
import static com.microsoft.ml.lightgbm.lightgbmlib.LGBM_BoosterSaveModelToStringSWIG;
import static com.microsoft.ml.lightgbm.lightgbmlib.delete_int64_tp;
import static com.microsoft.ml.lightgbm.lightgbmlib.new_int64_tp;
import static com.microsoft.ml.lightgbm.lightgbmlib.voidpp_value;
import static com.microsoft.ml.lightgbm.lightgbmlibConstants.C_API_FEATURE_IMPORTANCE_GAIN;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Native-allocation regression tests. They are opt-in because they depend on OS-specific committed-memory metrics
 * and on running against a freshly rebuilt native library.
 */
@EnabledIfSystemProperty(named = "lightgbm4j.nativeLeakTest", matches = "true")
public class NativeMemoryLeakTest {
    private static final int WARMUP_CALLS = 3;
    private static final int MEASURED_CALLS = 10;
    private static final int LARGE_MODEL_MEASURED_CALLS = 10;
    private static final int FAILURE_MEASURED_CALLS = 100;
    private static final int LARGE_MODEL_ITERATIONS = 30000;
    private static final long MODEL_SAVE_BUFFER_SIZE = 10L * 1024 * 1024;
    private static final long MAX_GROWTH_PER_CALL_BYTES = 2L * 1024 * 1024;

    @Test
    public void testSaveModelToStringDoesNotLeakNativeMemory() throws Exception {
        MemoryMetric memoryMetric = memoryMetric();
        Assumptions.assumeTrue(memoryMetric != null, "No supported committed-memory metric on this platform");

        LGBMDataset dataset = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "");
        try {
            long checksum = 0;
            for (int i = 0; i < WARMUP_CALLS; i++) {
                checksum += booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN).length();
            }
            System.gc();
            Thread.sleep(300);
            long baseline = memoryMetric.getCommittedBytes();

            for (int i = 0; i < MEASURED_CALLS; i++) {
                checksum += booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN).length();
            }
            System.gc();
            Thread.sleep(300);
            long after = memoryMetric.getCommittedBytes();

            long growthPerCall = (after - baseline) / MEASURED_CALLS;
            assertTrue(checksum > 0, "model strings should not be empty");
            assertTrue(growthPerCall < MAX_GROWTH_PER_CALL_BYTES,
                    "saveModelToString native memory growth per call should be below "
                            + MAX_GROWTH_PER_CALL_BYTES + " bytes, but was " + growthPerCall
                            + " bytes (baseline=" + baseline + ", after=" + after + ")");
        } finally {
            booster.close();
            dataset.close();
        }
    }

    @Test
    public void testDumpModelDoesNotLeakNativeMemory() throws Exception {
        MemoryMetric memoryMetric = memoryMetric();
        Assumptions.assumeTrue(memoryMetric != null, "No supported committed-memory metric on this platform");

        LGBMDataset dataset = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "");
        try {
            long checksum = 0;
            for (int i = 0; i < WARMUP_CALLS; i++) {
                checksum += dumpModel(booster).length();
            }
            System.gc();
            Thread.sleep(300);
            long baseline = memoryMetric.getCommittedBytes();

            for (int i = 0; i < MEASURED_CALLS; i++) {
                checksum += dumpModel(booster).length();
            }
            System.gc();
            Thread.sleep(300);
            long after = memoryMetric.getCommittedBytes();

            long growthPerCall = (after - baseline) / MEASURED_CALLS;
            assertTrue(checksum > 0, "model dumps should not be empty");
            assertTrue(growthPerCall < MAX_GROWTH_PER_CALL_BYTES,
                    "dumpModel native memory growth per call should be below "
                            + MAX_GROWTH_PER_CALL_BYTES + " bytes, but was " + growthPerCall
                            + " bytes (baseline=" + baseline + ", after=" + after + ")");
        } finally {
            booster.close();
            dataset.close();
        }
    }

    @Test
    public void testSaveModelToStringReallocationPathDoesNotLeakNativeMemory() throws Exception {
        MemoryMetric memoryMetric = memoryMetric();
        Assumptions.assumeTrue(memoryMetric != null, "No supported committed-memory metric on this platform");

        LGBMDataset dataset = LGBMDataset.createFromFile(
                "src/test/resources/cancer.csv",
                "header=true label=name:Classification",
                null
        );
        LGBMBooster booster = LGBMBooster.create(dataset,
                "objective=binary num_leaves=255 min_data_in_leaf=1 verbose=-1");
        try {
            for (int i = 0; i < LARGE_MODEL_ITERATIONS; i++) {
                booster.updateOneIter();
            }

            String model = booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
            assertTrue(model.length() > MODEL_SAVE_BUFFER_SIZE,
                    "test model should exceed the initial 10MB save buffer, but was "
                            + model.length() + " bytes");

            for (int i = 0; i < WARMUP_CALLS; i++) {
                booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
            }
            System.gc();
            Thread.sleep(300);
            long baseline = memoryMetric.getCommittedBytes();

            for (int i = 0; i < LARGE_MODEL_MEASURED_CALLS; i++) {
                booster.saveModelToString(0, 0, LGBMBooster.FeatureImportanceType.GAIN);
            }
            System.gc();
            Thread.sleep(300);
            long after = memoryMetric.getCommittedBytes();

            long growthPerCall = (after - baseline) / LARGE_MODEL_MEASURED_CALLS;
            assertTrue(growthPerCall < MAX_GROWTH_PER_CALL_BYTES,
                    "saveModelToString reallocation-path native memory growth per call should be below "
                            + MAX_GROWTH_PER_CALL_BYTES + " bytes, but was " + growthPerCall
                            + " bytes (baseline=" + baseline + ", after=" + after + ")");
        } finally {
            booster.close();
            dataset.close();
        }
    }

    @Test
    public void testSaveModelToStringFailurePathDoesNotLeakNativeMemory() throws Exception {
        MemoryMetric memoryMetric = memoryMetric();
        Assumptions.assumeTrue(memoryMetric != null, "No supported committed-memory metric on this platform");

        LGBMDataset dataset = LGBMDataset.createFromMat(new float[]{1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "", null);
        LGBMBooster booster = LGBMBooster.create(dataset, "");
        SWIGTYPE_p_long_long outLength = new_int64_tp();
        try {
            for (int i = 0; i < WARMUP_CALLS; i++) {
                assertNull(saveModelToStringWithInvalidFeatureImportance(booster, outLength));
            }
            System.gc();
            Thread.sleep(300);
            long baseline = memoryMetric.getCommittedBytes();

            for (int i = 0; i < FAILURE_MEASURED_CALLS; i++) {
                assertNull(saveModelToStringWithInvalidFeatureImportance(booster, outLength));
            }
            System.gc();
            Thread.sleep(300);
            long after = memoryMetric.getCommittedBytes();

            long growthPerCall = (after - baseline) / FAILURE_MEASURED_CALLS;
            assertTrue(growthPerCall < MAX_GROWTH_PER_CALL_BYTES,
                    "saveModelToString failure-path native memory growth per call should be below "
                            + MAX_GROWTH_PER_CALL_BYTES + " bytes, but was " + growthPerCall
                            + " bytes (baseline=" + baseline + ", after=" + after + ")");
        } finally {
            delete_int64_tp(outLength);
            booster.close();
            dataset.close();
        }
    }

    private String saveModelToStringWithInvalidFeatureImportance(LGBMBooster booster, SWIGTYPE_p_long_long outLength) throws Exception {
        return LGBM_BoosterSaveModelToStringSWIG(
                voidpp_value(boosterHandle(booster)),
                0,
                0,
                -1,
                MODEL_SAVE_BUFFER_SIZE,
                outLength
        );
    }

    private String dumpModel(LGBMBooster booster) throws Exception {
        SWIGTYPE_p_long_long outLength = new_int64_tp();
        try {
            return LGBM_BoosterDumpModelSWIG(
                    voidpp_value(boosterHandle(booster)),
                    0,
                    0,
                    C_API_FEATURE_IMPORTANCE_GAIN,
                    10L * 1024 * 1024,
                    outLength
            );
        } finally {
            delete_int64_tp(outLength);
        }
    }

    private SWIGTYPE_p_p_void boosterHandle(LGBMBooster booster) throws Exception {
        Field handle = LGBMBooster.class.getDeclaredField("handle");
        handle.setAccessible(true);
        return (SWIGTYPE_p_p_void) handle.get(booster);
    }

    private MemoryMetric memoryMetric() {
        String os = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        if (os.contains("linux")) {
            return new LinuxVmSizeMetric();
        }
        if (os.contains("windows")) {
            return WindowsCommittedMemoryMetric.create();
        }
        return null;
    }

    private interface MemoryMetric {
        long getCommittedBytes() throws Exception;
    }

    private static class LinuxVmSizeMetric implements MemoryMetric {
        @Override
        public long getCommittedBytes() throws Exception {
            List<String> status = Files.readAllLines(Paths.get("/proc/self/status"));
            for (String line : status) {
                if (line.startsWith("VmSize:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
                }
            }
            throw new IllegalStateException("VmSize not found in /proc/self/status");
        }
    }

    private static class WindowsCommittedMemoryMetric implements MemoryMetric {
        private final Object osMxBean;
        private final Method committedVirtualMemorySize;

        private WindowsCommittedMemoryMetric(Object osMxBean, Method committedVirtualMemorySize) {
            this.osMxBean = osMxBean;
            this.committedVirtualMemorySize = committedVirtualMemorySize;
        }

        static WindowsCommittedMemoryMetric create() {
            try {
                Object osMxBean = ManagementFactory.getOperatingSystemMXBean();
                Class<?> mxBeanInterface = Class.forName("com.sun.management.OperatingSystemMXBean");
                if (!mxBeanInterface.isInstance(osMxBean)) {
                    return null;
                }
                Method method = mxBeanInterface.getMethod("getCommittedVirtualMemorySize");
                return new WindowsCommittedMemoryMetric(osMxBean, method);
            } catch (ReflectiveOperationException | SecurityException e) {
                return null;
            }
        }

        @Override
        public long getCommittedBytes() throws Exception {
            return (Long) committedVirtualMemorySize.invoke(osMxBean);
        }
    }
}
