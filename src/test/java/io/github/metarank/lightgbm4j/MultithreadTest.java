package io.github.metarank.lightgbm4j;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class MultithreadTest {
    @Test
    public void testThreading() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(32);
        List<CreateTask> queue = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            queue.add(new CreateTask());
        }
        pool.invokeAll(queue);
    }


    public class CreateTask implements Callable<Integer> {

        @Override
        public Integer call() throws Exception {
            LGBMDataset ds = LGBMDataset.createFromMat(new float[] {1.0f, 1.0f, 1.0f, 1.0f}, 2, 2, true, "");
            LGBMBooster.create(ds, "");

            return 0;
        }
    }
}
