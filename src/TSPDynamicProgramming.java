import java.io.FileWriter;
import java.io.PrintWriter;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class TSPDynamicProgramming {

        static ThreadMXBean bean = ManagementFactory.getThreadMXBean( );

        /* define constants */
        static long MAXVALUE =  2000000000;
        static long MINVALUE = -2000000000;
        static int numberOfTrials = 20;
        static int MAXINPUTSIZE  = (int) Math.pow(1.5,28);
        static int MININPUTSIZE  =  1;
        static int Nums = 20;
        static long fibResult = 0;
        // static int SIZEINCREMENT =  10000000; // not using this since we are doubling the size each time

        static String ResultsFolderPath = "/home/diana/Results/"; // pathname to results folder
        static FileWriter resultsFile;
        static PrintWriter resultsWriter;

        static double result;

    public static class PathAndMatrix{
            int[] travelNodes;
            int[][] costMatrix;
        }

        public static class PathAndCost{
            int[] path;
            int cost;
        }

        public static class vertex {
            String name;
            int x;
            int y;
        }

        public static void main(String[] args) {

            // run the whole experiment at least twice, and expect to throw away the data from the earlier runs, before java has fully optimized

            System.out.println("Running first full experiment...");
            runFullExperiment("TSPDynamic-Exp1-ThrowAway.txt");
            System.out.println("Running second full experiment...");
            runFullExperiment("TSPDynamic-Exp2.txt");
            System.out.println("Running third full experiment...");
            runFullExperiment("TSPDynamic-Exp3.txt");
        }

        static void runFullExperiment(String resultsFileName){
            //declare variables for doubling ratio
            double[] averageArray = new double[1000];
            double currentAv = 0;
            double doublingTotal = 0;
            int x = 0;
            int angle = 70;
            int r = 12;
            //int n = 360/angle;

            //int n = 5;
            //double[][] costMatrix =  GenerateRandomCircularGraphCostMatrix(n, r, angle);
            //double[][] costMatrix = GenerateRandomCostMatrix(n);

            // If the array is empty
            // or the index is not in array range
            // return the original array
            //PathMatrix.travelNodes = removeElements(PathMatrix.travelNodes);

            //TspDynamicProgrammingRecursive solver = new TspDynamicProgrammingRecursive(costMatrix);



            //System.out.println("Best Cost = " + solver.getTourCost());
            //System.out.println("Path = " + solver.getTour());


            //GenerateRandomCostMatrix(5);
            //GenerateRandomEuclideanCostMatrix(5);
            //set up print to file
            try {
                resultsFile = new FileWriter(ResultsFolderPath + resultsFileName);
                resultsWriter = new PrintWriter(resultsFile);
            } catch(Exception e) {
                System.out.println("*****!!!!!  Had a problem opening the results file "+ResultsFolderPath+resultsFileName);
                return; // not very foolproof... but we do expect to be able to create/open the file...
            }

            //declare variables for stop watch
            ThreadCpuStopWatch BatchStopwatch = new ThreadCpuStopWatch(); // for timing an entire set of trials
            ThreadCpuStopWatch TrialStopwatch = new ThreadCpuStopWatch(); // for timing an individual trial

            //add headers to text file
            resultsWriter.println("#X(Value)  N(Size)  AverageTime        FibNumber   NumberOfTrials"); // # marks a comment in gnuplot data
            resultsWriter.flush();

            /* for each size of input we want to test: in this case starting small and doubling the size each time */
            for(int inputSize=3;inputSize<=Nums; inputSize++) {
                //test run for fibonacci numbers
                //verifyGreedySalesman(inputSize);
                double[][] costMatrix = GenerateRandomEuclideanCostMatrix(inputSize);
                verifyTSPDynamic(inputSize, costMatrix);
                // progress message...
                System.out.println("Running test for input size "+inputSize+" ... ");

                /* repeat for desired number of trials (for a specific size of input)... */
                long batchElapsedTime = 0;
                // generate a list of randomly spaced integers in ascending sorted order to use as test input
                // In this case we're generating one list to use for the entire set of trials (of a given input size)
                // but we will randomly generate the search key for each trial
                //System.out.print("    Generating test data...");

                //generate random integer list
                //long resultFib = Fib(x);
                double[][] costMatrix1 = GenerateRandomEuclideanCostMatrix(inputSize);
                //print progress to screen
                //System.out.println("...done.");
                System.out.print("    Running trial batch...");

                /* force garbage collection before each batch of trials run so it is not included in the time */
                System.gc();


                // instead of timing each individual trial, we will time the entire set of trials (for a given input size)
                // and divide by the number of trials -- this reduces the impact of the amount of time it takes to call the
                // stopwatch methods themselves
                BatchStopwatch.start(); // comment this line if timing trials individually

                // run the trials
                for (long trial = 0; trial < numberOfTrials; trial++) {
                    // generate a random key to search in the range of a the min/max numbers in the list
                    //long testSearchKey = (long) (0 + Math.random() * (testList[testList.length-1]));
                    /* force garbage collection before each trial run so it is not included in the time */
                    // System.gc();

                    //TrialStopwatch.start(); // *** uncomment this line if timing trials individually
                    /* run the function we're testing on the trial input */
                    TspDynamicProgrammingRecursive solver = new TspDynamicProgrammingRecursive(costMatrix1);
                    result = solver.getTourCost();

                    //fibResult = Greedy(inputSize);
                    //System.out.println(result);
                    // batchElapsedTime = batchElapsedTime + TrialStopwatch.elapsedTime(); // *** uncomment this line if timing trials individually
                }
                batchElapsedTime = BatchStopwatch.elapsedTime(); // *** comment this line if timing trials individually
                double averageTimePerTrialInBatch = (double) batchElapsedTime / (double)numberOfTrials; // calculate the average time per trial in this batch

                //put current average time in array of average times. We will be able to use this to calculate the doubling ratio
                averageArray[x] = averageTimePerTrialInBatch;

                //skip this round if this is the first one (no previous average for calculation)
                if(inputSize != 0){
                    //doublingTotal = averageTimePerTrialInBatch/averageArray[x-1]; //Calculate doubling ratio

                }
                x++;
                int countingbits = countBits(inputSize);

                /* print data for this size of input */
                resultsWriter.printf("%6d %6d %15.2f %10.2f %4d\n",inputSize, countingbits, averageTimePerTrialInBatch, result, numberOfTrials); // might as well make the columns look nice
                resultsWriter.flush();
                System.out.println(" ....done.");
            }
        }

        /*Verify merge sort is working*/
        static void verifyTSPDynamic(int x, double[][] costMatrix){

            System.out.println("Testing..." + x);
            TspDynamicProgrammingRecursive solver = new TspDynamicProgrammingRecursive(costMatrix);
            System.out.println("Cost = " + solver.getTourCost());
            System.out.println("Path = " + solver.getTour());
        }

        //Remove first and last elements from travelNodes
        static int[] removeElements(int[] travelNodes){
            int index = 0;
            int[] tempNodes = new int[travelNodes.length-1];

            for(int i = 0, k = 0; i < travelNodes.length; i++){
                if(i == index){
                    continue;
                }

                // if the index is not
                // the removal element index
                tempNodes[k++] = travelNodes[i];
            }
            travelNodes = tempNodes;

            index = travelNodes.length-1;
            tempNodes = new int[travelNodes.length-1];

            for(int i = 0, k = 0; i < travelNodes.length; i++){
                if(i == index){
                    continue;
                }

                // if the index is not
                // the removal element index
                tempNodes[k++] = travelNodes[i];
            }

            travelNodes = tempNodes;

            return travelNodes;
        }

        public static double[][] GenerateRandomCostMatrix(int n){
            //declare variables
            double[][] randomCostMatrix = new double[n][n];
            int halfN = n-Math.abs(n/2);
            int num = 1;
            //generate matrix - loop through the matrix and assign random numbers as the cost
            for(int t = 0; t < n; t++){
                for(int q =num; q < n; q++) {
                    if(t == q){
                        randomCostMatrix[t][q] = 0;
                    }
                    else{
                        randomCostMatrix[t][q] = (int) (Math.random()*20 + 1) + 1;
                        randomCostMatrix[q][t] = randomCostMatrix[t][q];
                    }
                }
                num = num + 1;
            }
            //System.out.println(Arrays.deepToString(randomCostMatrix));

            //return the matrix
            return  randomCostMatrix;
        }

        public static double[][] GenerateRandomEuclideanCostMatrix(int n){
            //declare variables
            double[][] randomEuclideanCostMatrix = new double[n][n];
            vertex[] v = new vertex[n];
            for(int s = 0; s<v.length; s++){
                v[s] = new vertex();
            }
            int maxX = 100; //the max value for x and y coordinates
            int maxY = 100;
            //calculate random values for the x,y coordinates of the vertices
            for(int i =0; i < n; i++) {
                v[i].name = Integer.toString(i);
                v[i].x = (int) (Math.random() * (maxX + 1) + 1);
                v[i].y = (int) (Math.random() * (maxY + 1) + 1);
                //System.out.println(v[i].x + "," + v[i].y);
            }
            //calculate the distance between vertices using their x,y coordinates
            for(int t = 0; t < n; t++){
                for(int q = 0; q < n; q++){
                    randomEuclideanCostMatrix[t][q] = (int) Math.sqrt(Math.pow(v[t].x - v[q].x,2) +
                            Math.pow(v[t].y - v[q].y, 2) *1);
                }
            }
           // System.out.println(Arrays.deepToString(randomEuclideanCostMatrix));

            //return th matrix
            return  randomEuclideanCostMatrix;
        }

        public static double[][] GenerateRandomCircularGraphCostMatrix(int n, int r, int angle){
            //declare variables
            double[][] randomCircularCostMatrix = new double[n][n];
            //int[] vertexList = new int[n+1];
            vertex[] v = new vertex[n+1];
            vertex[] v2 = new vertex[n];
            int angle2 = 0;
            //initialize the array of vertices
            for(int s = 0; s<v.length; s++){
                v[s] = new vertex();
            }
            //generate random x,y coordinates for each vertex
            for(int i =0; i < n; i++) {
                angle2 = angle2 + angle;
                v[i].name = Integer.toString(i);
                v[i].x = Math.abs((int) ( r * Math.cos(angle2)));
                v[i].y = Math.abs((int) (r * Math.sin(angle2)));
                //System.out.println(v[i].x + "," + v[i].y);
            }
            //randomize the array
            Random rnd = ThreadLocalRandom.current();
            for (int i = v.length - 2; i > 0; i--){
                int index = rnd.nextInt(i + 1);
                if(index == 0){

                }
                else{
                    // Simple swap
                    vertex a = v[index];
                    v[index] = v[i];
                    v[i] = a;
                }
            }
            v[n].x = 0;
            v[n].y = 0;
            v[n].name = "0";

            for(int i = 0; i<v.length-1; i++) {
                //System.out.println(v[i].name + ',' + v[i].x + "," + v[i].y);
            }
            //assign 0 to each cost
            for(int c = 0; c<v.length-1; c++) {
                for(int t = 0; t<v.length-1; t++) {

                        randomCircularCostMatrix[c][t] = 0;
                        randomCircularCostMatrix[t][c] = 0;
                }
            }

            /*for(int t= 0; t<v.length; t++) {
                for (int q = t + 1; q < v.length; q++) {

                }
            }*/
            //calculate the distance between adjacent vertices
            for(int t = 0; t<v.length-1; t++){
                randomCircularCostMatrix[Integer.parseInt((v[t].name))][Integer.parseInt(v[t+1].name)] = (int) Math.sqrt(Math.pow(v[t].x - v[t+1].x,2) +
                        Math.pow(v[t].y - v[t+1].y, 2) *1);
                randomCircularCostMatrix[Integer.parseInt((v[t+1].name))][Integer.parseInt(v[t].name)] =  randomCircularCostMatrix[Integer.parseInt((v[t].name))][Integer.parseInt(v[t+1].name)];
            }
            //System.out.println(Arrays.deepToString(randomCircularCostMatrix));

            int[] tempArray = new int[v.length];

            for(int i = 0; i< v.length-1; i++){
                tempArray[i] = Integer.parseInt(v[i].name);
            }

            //System.out.println(Arrays.deepToString(randomCircularCostMatrix));
            //return matrix
            return  randomCircularCostMatrix;
        }

    public static class TspDynamicProgrammingRecursive {

        private final int N;
        private final int START_NODE;
        private final int FINISHED_STATE;

        private double[][] distance;
        private double minTourCost = Double.POSITIVE_INFINITY;

        private List<Integer> tour = new ArrayList<>();
        private boolean ranSolver = false;

        public TspDynamicProgrammingRecursive(double[][] distance) {
            this(0, distance);
        }

        public TspDynamicProgrammingRecursive(int startNode, double[][] distance) {

            this.distance = distance;
            N = distance.length;
            START_NODE = startNode;

            // Validate inputs.
            if (N <= 2) throw new IllegalStateException("TSP on 0, 1 or 2 nodes doesn't make sense.");
            if (N != distance[0].length)
                throw new IllegalArgumentException("Matrix must be square (N x N)");
            if (START_NODE < 0 || START_NODE >= N)
                throw new IllegalArgumentException("Starting node must be: 0 <= startNode < N");
            if (N > 32)
                throw new IllegalArgumentException(
                        "Matrix too large! A matrix that size for the DP TSP problem with a time complexity of"
                                + "O(n^2*2^n) requires way too much computation for any modern home computer to handle");

            // The finished state is when the finished state mask has all bits are set to
            // one (meaning all the nodes have been visited).
            FINISHED_STATE = (1 << N) - 1;
        }

        // Returns the optimal tour for the traveling salesman problem.
        public List<Integer> getTour() {
            if (!ranSolver) solve();
            return tour;
        }

        // Returns the minimal tour cost.
        public double getTourCost() {
            if (!ranSolver) solve();
            return minTourCost;
        }

        public void solve() {

            // Run the solver
            int state = 1 << START_NODE;
            Double[][] memo = new Double[N][1 << N];
            Integer[][] prev = new Integer[N][1 << N];
            minTourCost = tsp(START_NODE, state, memo, prev);

            // Regenerate path
            int index = START_NODE;
            while (true) {
                tour.add(index);
                Integer nextIndex = prev[index][state];
                if (nextIndex == null) break;
                int nextState = state | (1 << nextIndex);
                state = nextState;
                index = nextIndex;
            }
            tour.add(START_NODE);
            ranSolver = true;
        }

        private double tsp(int i, int state, Double[][] memo, Integer[][] prev) {

            // Done this tour. Return cost of going back to start node.
            if (state == FINISHED_STATE) return distance[i][START_NODE];

            // Return cached answer if already computed.
            if (memo[i][state] != null) return memo[i][state];

            double minCost = Double.POSITIVE_INFINITY;
            int index = -1;
            for (int next = 0; next < N; next++) {

                // Skip if the next node has already been visited.
                if ((state & (1 << next)) != 0) continue;

                int nextState = state | (1 << next);
                if(distance[i][next] != 0){
                    double newCost = distance[i][next] + tsp(next, nextState, memo, prev);
                    if (newCost < minCost) {
                        minCost = newCost;
                        index = next;
                    }
                }

            }

            prev[i][state] = index;
            return memo[i][state] = minCost;
        }
    }

        //count the number of bits required for current fib number
        static int countBits(int n) {
            int count = 0;
            //if n == 0, count will be 1
            if (n == 0) {
                count = 1;
            }
            //loop while n does not equal 0
            while (n != 0) {
                //each loop add 1 to count
                count++;
                //shift n to the left by 1
                n >>= 1;
            }
            //System.out.println("number of bits = " + count);
            return count;
        }


}
