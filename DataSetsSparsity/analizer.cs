using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using Amazon.S3.IO;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Math.Decompositions;
using Accord.Statistics;
using Accord.Statistics.Analysis;

namespace DataSetsSparsity
{
    class analizer
    {
        private string analysisFolderName;
        private List<List<double>> MainGrid;
        private DB db;
        private recordConfig rc;

        public analizer(string analysisFolderName, List<List<double>> MainGrid, DB db, recordConfig rc)
        {
            // TODO: Complete member initialization
            this.analysisFolderName = analysisFolderName;
            this.MainGrid = MainGrid;
            this.db = db;
            this.rc = rc;


        }

        public void analize(List<int> trainingArr, List<int> testingArr, int[][] boundingBox)
        {
            #region general

            var watch = Stopwatch.StartNew();
            int WGBseed = 9999;
            int GBseed = WGBseed;

            //CREATE DECISION TREES

            //RAND DIM
            bool[] Dim2TakeOneTree = getDim2Take(rc, 1); //take all
            decicionTree decTree = new decicionTree(rc, db, Dim2TakeOneTree);// יוצר מבנה נתונים של עץ, אבל עדיין אין עץ עם עלים וכו

            //decicionTree decTree = new decicionTree(rc, db);
            List<GeoWave> decision_GeoWaveArr = decTree.getdecicionTree(trainingArr, boundingBox); // יוצר את עץ ההחלטה כולו
            watch.Stop();

            double[] toc_time = new double[1];
            toc_time[0] = watch.ElapsedMilliseconds;

            // עד כאן רלוונטי גם לשאר המקרים  - לא רק עץ יחיד

            printErrorsOfTree(toc_time, analysisFolderName + "\\time_to_generate_FullTree.txt");

            double[] nWaev = new double[1];
            nWaev[0] = decision_GeoWaveArr.Count;
            printErrorsOfTree(nWaev, analysisFolderName + "\\NwaveletsInTree.txt");

            List<GeoWave> final_GeoWaveArr = decision_GeoWaveArr.OrderByDescending(o => o.norm).ToList();//see if not sorted by norm already...

            int testBegin = rc.waveletsTestRange[0];
            int arrSize = rc.waveletsTestRange[1] == 0 ? Convert.ToInt32(rc.test_error_size * final_GeoWaveArr.Count / rc.hopping_size) : Convert.ToInt32((1 + rc.waveletsTestRange[1] - rc.waveletsTestRange[0]) / rc.hopping_size);

            double[] errorTree = new double[arrSize];
            double[] decayOnTraining = new double[arrSize];
            double[] errorTreeL1 = new double[arrSize];
            double[] errorTreeBER = new double[arrSize];
            double[] missLabels = new double[arrSize];
            double[] Nwavelets = new double[arrSize];
            int LPnorm2Error = (rc.DataType.Contains("regression")) ? 2 : -10;
            List<int> validationArr = new List<int>();
            List<int> OriginaltrainingArr = trainingArr;

            if (rc.percent_training_db < 1)
            {
                int validLength = Convert.ToUInt16(Math.Floor((1 - rc.percent_training_db) * trainingArr.Count()));
                validationArr = trainingArr.GetRange(0, validLength);
                trainingArr = new List<int>();
                trainingArr = OriginaltrainingArr.GetRange(validLength, OriginaltrainingArr.Count() - validLength).ToList();
            }
            //TODO: SEND THIS PARAMETERS FROM GUI.
            double bag_fraction = 0.8;
            #endregion

            #region one tree

            if (Form1.u_config.runOneTreeCB == "1")
            {
                if (Form1.rumPrallel)
                {
                    Parallel.For(testBegin, arrSize, i =>
                    {
                        errorTree[i] = testDecisionTree(testingArr, db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, rc.NormLPType);
                        if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                            decayOnTraining[i] = testDecisionTree(trainingArr, db.PCAtraining_dt, db.training_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, rc.NormLPType);
                        Nwavelets[i] = i * rc.hopping_size;
                    });
                }
                else
                {
                    for (int i = testBegin; i < arrSize; i++)
                    {
                        //double dbgNorm = 0;
                        errorTree[i] = testDecisionTree(testingArr, db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, rc.NormLPType);
                        if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                            decayOnTraining[i] = testDecisionTree(trainingArr, db.PCAtraining_dt, db.training_label, decision_GeoWaveArr, final_GeoWaveArr[i * rc.hopping_size].norm, rc.NormLPType);
                        Nwavelets[i] = i * rc.hopping_size;
                    }
                }

                int minErr_index = Enumerable.Range(0, errorTree.Length).Aggregate((a, b) => (errorTree[a] < errorTree[b]) ? a : b); //minerror
                double lowest_Tree_error = testDecisionTree(testingArr, db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, final_GeoWaveArr[minErr_index * rc.hopping_size].norm, rc.NormLPType);
                printErrorsOfTree(lowest_Tree_error, minErr_index * rc.hopping_size, analysisFolderName + "\\bsp_tree_errors_by_wavelets_TestDB.txt");

                //PRINT ERRORS TO FILE...
                printErrorsOfTree(errorTree, Nwavelets, analysisFolderName + "\\bsp_tree_errors_by_wavelets_ValidationDB.txt");
                if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                    printErrorsOfTree(decayOnTraining, Nwavelets, analysisFolderName + "\\bsp_tree_errors_by_wavelets_trainingDB.txt");

            }

            #region prooning one tree            

            if (Form1.runProoning)
            {
                //TEST TREE WITH PROONING
                int topLevelBegin = rc.pruningTestRange[0];
                int topLevel = rc.waveletsTestRange[1] == 0 ? getTopLevel(decision_GeoWaveArr) : rc.waveletsTestRange[1];

                //int topLevel = getTopLevel(decision_GeoWaveArr);
                double[] errorTreeProoning = new double[topLevel];
                double[] errorTreeProoningOnTraining = new double[topLevel];
                //double[] errorTreeProoningL1 = new double[topLevel];
                double[] NLevels = new double[topLevel];
                //double[] errorTreeProoningBER = new double[topLevel];

                if (Form1.rumPrallel)
                {
                    Parallel.For(topLevelBegin, topLevel, i =>
                    {
                        errorTreeProoning[i] = testDecisionTreeWithProoning(testingArr, db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, i + 1, rc.NormLPType);
                        if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                            errorTreeProoningOnTraining[i] = testDecisionTreeWithProoning(db.PCAtraining_dt, db.training_label, decision_GeoWaveArr, i + 1, rc.NormLPType);
                        //errorTreeProoningBER[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, -2);
                        //errorTreeProoningL1[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, 1);
                        NLevels[i] = i;// * rc.hopping_size;
                    });
                }
                else
                {
                    for (int i = topLevelBegin; i < topLevel; i++)
                    {
                        errorTreeProoning[i] = testDecisionTreeWithProoning(testingArr, db.PCAvalidation_dt, db.validation_label, decision_GeoWaveArr, i + 1, rc.NormLPType);
                        if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                            errorTreeProoningOnTraining[i] = testDecisionTreeWithProoning(db.PCAtraining_dt, db.training_label, decision_GeoWaveArr, i + 1, rc.NormLPType);
                        //errorTreeProoningBER[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, -2);
                        //errorTreeProoningL1[i] = testDecisionTreeWithProoning(db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, i + 1, 1);
                        NLevels[i] = i;// *rc.hopping_size;
                    }
                }


                int minErrPruning_index = Enumerable.Range(0, errorTreeProoning.Length).Aggregate((a, b) => (errorTreeProoning[a] < errorTreeProoning[b]) ? a : b); //minerror
                double lowest_TreePruning_error = testDecisionTree(testingArr, db.PCAtesting_dt, db.testing_label, decision_GeoWaveArr, minErrPruning_index + 1, rc.NormLPType);
                printErrorsOfTree(lowest_TreePruning_error, minErrPruning_index, analysisFolderName + "\\bsp_tree_errors_by_waveletsPruning_TestDB.txt");

                //PRINT ERRORS TO FILE...
                printErrorsOfTree(errorTreeProoning, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooning_Validation.txt");
                if (Form1.u_config.runOneTreeOnTtrainingCB == "1")
                    printErrorsOfTree(errorTreeProoningOnTraining, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooning_training.txt");
                //printErrorsOfTree(errorTreeProoningBER, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooningBER.txt");
                //printErrorsOfTree(errorTreeProoningL1, NLevels, analysisFolderName + "\\bsp_tree_errors_by_prooningL1.txt");
                #endregion

            }

            #endregion

            #region RF tree 

            int tmp_N_rows = Convert.ToInt32(trainingArr.Count * rc.rfBaggingPercent);
            //List<int>[] trainingArrRF_indecesList = new List<int>[tmp_N_rows];
            List<int>[] trainingArrRF_indecesList = new List<int>[rc.rfNum];


            if (Form1.runRf)
            {
                //create RF
                List<GeoWave>[] RFdecTreeArr = new List<GeoWave>[rc.rfNum];

                if (Form1.rumPrallel)
                {
                    Parallel.For(0, rc.rfNum, i =>
                    {
                        List<int> trainingArrRF;
                        if (Form1.u_config.BaggingWithRepCB == "1")
                            trainingArrRF = BaggingBreiman(trainingArr, i);
                        else
                            trainingArrRF = Bagging(trainingArr, rc.rfBaggingPercent, i);

                        trainingArrRF_indecesList[i] = trainingArrRF;
                        bool[] Dim2Take = getDim2Take(rc, i);
                        decicionTree decTreeRF = new decicionTree(rc, db, Dim2Take);
                        RFdecTreeArr[i] = decTreeRF.getdecicionTree(trainingArrRF, boundingBox, i);
                    });
                }
                else
                {
                    for (int i = 0; i < rc.rfNum; i++)
                    {
                        List<int> trainingArrRF;
                        if (Form1.u_config.BaggingWithRepCB == "1")
                            trainingArrRF = BaggingBreiman(trainingArr, i);
                        else
                            trainingArrRF = Bagging(trainingArr, rc.rfBaggingPercent, i);

                        trainingArrRF_indecesList[i] = trainingArrRF;
                        bool[] Dim2Take = getDim2Take(rc, i);
                        decicionTree decTreeRF = new decicionTree(rc, db, Dim2Take);
                        //decicionTree decTreeRF = new decicionTree(rc, db);
                        //RFdecTreeArr[i] = decTree.getdecicionTree(trainingArrRF, boundingBox);
                        RFdecTreeArr[i] = decTreeRF.getdecicionTree(trainingArrRF, boundingBox, i);
                        //Form1.printConstWavelets2File(RFdecTreeArr[i], analysisFolderName + "\\RFdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                        //Form1.printtable(db.PCAtraining_dt, analysisFolderName + "\\PCA_DATA_" + i.ToString() + "tree.txt", trainingArrRF);
                        //Form1.printtable(db.training_label, analysisFolderName + "\\PCA_label_" + i.ToString() + "tree.txt", trainingArrRF);
                    }
                }

                //sparse the forest to have max "1000" wavelets in each tree
                if (Form1.u_config.sparseRfCB == "1" && Form1.u_config.sparseRfTB != "")
                {
                    int NwaveletsTmp;
                    if (int.TryParse(Form1.u_config.sparseRfTB, out NwaveletsTmp))
                        RFdecTreeArr = getsparseRF(RFdecTreeArr, NwaveletsTmp);
                }



                if (Form1.u_config.saveTressCB == "1")
                {
                    if (!System.IO.Directory.Exists(analysisFolderName + "\\archive"))
                        System.IO.Directory.CreateDirectory(analysisFolderName + "\\archive");
                    //Form1.printtable(trainingArrRF_indecesList, analysisFolderName + "\\RFIndeces.txt"); //- not for giant DB
                    for (int i = 0; i < RFdecTreeArr.Count(); i++)
                    {
                        Form1.printWaveletsProperties(RFdecTreeArr[i], analysisFolderName + "\\archive\\waveletsPropertiesTree_" + i.ToString() + ".txt");
                        //Form1.printConstWavelets2File(RFdecTreeArr[i], analysisFolderName + "\\archive\\RFdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                    }
                }

                List<double> NormArr = new List<double>();
                //for (int j = 0; j < RFdecTreeArr.Count(); j++)
                for (int j = 0; j < 1; j++)//go over tree j==0 (first)
                    for (int i = 0; i < RFdecTreeArr[j].Count; i++)
                    {
                        //if (RFdecTreeArr[j][i].level <= rc.BoundLevel)//restrict the level we take wavelets from
                        NormArr.Add(RFdecTreeArr[j][i].norm);
                    }

                if (Form1.u_config.estimateRFwaveletsCB == "1")
                {
                    ////int arrSizeRF = Convert.ToInt32(rc.test_error_size * NormArr.Count / rc.hopping_size);
                    int RFtestBegin = rc.RFwaveletsTestRange[0];
                    int arrSizeRF = rc.RFwaveletsTestRange[1] == 0 ? Convert.ToInt32(rc.test_error_size * NormArr.Count() / rc.hopping_size) : Convert.ToInt32((1 + rc.RFwaveletsTestRange[1] - rc.RFwaveletsTestRange[0]) / rc.hopping_size);

                    //int arrSizeRF = 100;//DBG!!!!

                    //double[] errorRF = new double[arrSizeRF];
                    //double[] errorManyRF = new double[arrSizeRF];
                    double[][] decayManyRF = new double[arrSizeRF][];
                    double[][] errorManyRF = new double[arrSizeRF][];
                    double[][] errorManyRFNoVoting = new double[arrSizeRF][];
                    double[][] errorManyRFNoVoting_training = new double[arrSizeRF][];

                    for (int i = 0; i < arrSizeRF; i++)
                    {
                        errorManyRF[i] = new double[RFdecTreeArr.Count()];
                        decayManyRF[i] = new double[RFdecTreeArr.Count()];
                        errorManyRFNoVoting[i] = new double[RFdecTreeArr.Count()];
                        errorManyRFNoVoting_training[i] = new double[RFdecTreeArr.Count()];
                    }

                    //double[] missLabelsRF = new double[arrSizeRF];
                    double[] NwaveletsRF = new double[arrSizeRF];


                    if (Form1.rumPrallel)
                    {
                        Parallel.For(0, arrSizeRF, i =>
                        {
                            //errorRF[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                            //errorRF[i] = testDecisionTreeManyRFNormNbound(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], rc.BoundLevel, 2);
                            errorManyRF[i] = testDecisionTreeManyRF(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            if (Form1.u_config.estimateRFonTrainingCB == "1")
                                decayManyRF[i] = testDecisionTreeManyRF(db.PCAtraining_dt, db.training_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            if (Form1.u_config.estimateRFnoVotingCB == "1")
                                errorManyRFNoVoting[i] = testDecisionTreeManyRFNoVoting(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            if (Form1.u_config.estimateRFnoVotingCB == "1" && Form1.u_config.estimateRFonTrainingCB == "1")
                                errorManyRFNoVoting_training[i] = testDecisionTreeManyRFNoVoting(trainingArrRF_indecesList, db.PCAtraining_dt, db.training_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            //missLabelsRF[i] = testDecisionTreeRF(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], 0);
                            NwaveletsRF[i] = RFtestBegin + i * rc.hopping_size;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                        });
                    }
                    else
                    {
                        for (int i = 0; i < arrSizeRF; i++)
                        {
                            //errorRF[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], 2);
                            //errorRF[i] = testDecisionTreeManyRFNormNbound(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], rc.BoundLevel, 2);
                            errorManyRF[i] = testDecisionTreeManyRF(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            if (Form1.u_config.estimateRFonTrainingCB == "1")
                                decayManyRF[i] = testDecisionTreeManyRF(db.PCAtraining_dt, db.training_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            if (Form1.u_config.estimateRFnoVotingCB == "1")
                                errorManyRFNoVoting[i] = testDecisionTreeManyRFNoVoting(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            if (Form1.u_config.estimateRFnoVotingCB == "1" && Form1.u_config.estimateRFonTrainingCB == "1")
                                errorManyRFNoVoting_training[i] = testDecisionTreeManyRFNoVoting(trainingArrRF_indecesList, db.PCAtraining_dt, db.training_label, RFdecTreeArr, NormArr[RFtestBegin + i * rc.hopping_size], rc.NormLPType);
                            //missLabelsRF[i] = testDecisionTreeRF(db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[i * rc.hopping_size], 0);
                            NwaveletsRF[i] = RFtestBegin + i * rc.hopping_size;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                        }
                    }

                    ////int minErrRF_index = Enumerable.Range(0, errorRF.Length).Aggregate((a, b) => (errorRF[a] < errorRF[b]) ? a : b); //minerror
                    ////int minErrRF_index = Enumerable.Range(0, errorManyRF.Length).Aggregate((a, b) => (errorManyRF[a][rc.rfNum - 1] < errorManyRF[b][rc.rfNum - 1]) ? a : b); //minerror
                    ////double lowest_TreeRF_error = testDecisionTreeRF(testingArr, db.PCAtesting_dt, db.testing_label, RFdecTreeArr, NormArr[RFtestBegin + minErrRF_index * rc.hopping_size], 2);
                    ////printErrorsOfTree(lowest_TreeRF_error, minErrRF_index * rc.hopping_size, analysisFolderName + "\\bsp_tree_errors_by_wavelets_RF_TestDB.txt");


                    ////printErrorsOfTree(errorRF, NwaveletsRF, analysisFolderName + "\\RF_errors_by_waveletsbounded_at_level_" + rc.BoundLevel.ToString() + ".txt");
                    ////printErrorsOfTree(errorRF, NwaveletsRF, analysisFolderName + "\\RF_errors_by_wavelets_validationDB.txt");


                    Form1.printtable(errorManyRF, analysisFolderName + "\\cumulative_RF_errors_by_wavelets_norm_threshold_validation.txt");
                    if (Form1.u_config.estimateRFonTrainingCB == "1")
                        Form1.printtable(decayManyRF, analysisFolderName + "\\cumulative_RF_errors_on_training_by_wavelets_norm_threshold.txt");
                    if (Form1.u_config.estimateRFnoVotingCB == "1")
                        Form1.printtable(errorManyRFNoVoting, analysisFolderName + "\\independent_errors_of_rf_treees_no_voting_wavelets_norm_threshold_validation.txt");
                    if (Form1.u_config.estimateRFnoVotingCB == "1" && Form1.u_config.estimateRFonTrainingCB == "1")
                        Form1.printtable(errorManyRFNoVoting_training, analysisFolderName + "\\independent_errors_of_rf_treees_no_voting_wavelets_norm_threshold_training.txt");

                    //////tmp - for smoothness
                    //////*******************************************************************************
                    ////Form1.printtable(no_smootherrorManyRF, analysisFolderName + "\\cumulative_no_smootherrorManyRF_errors_by_wavelets_norm_threshold_validation.txt");
                    ////Form1.printtable(medium_smootherrorManyRF, analysisFolderName + "\\cumulative_medium_smootherrorManyRF_errors_by_wavelets_norm_threshold_validation.txt");
                    ////Form1.printtable(high_smootherrorManyRF, analysisFolderName + "\\high_smootherrorManyRF_errors_by_wavelets_norm_threshold_validation.txt");
                    ////Form1.printtable(mix_smootherrorManyRF, analysisFolderName + "\\mix_smootherrorManyRF_errors_by_wavelets_norm_threshold_validation.txt");
                    //////tmp - for smoothness
                    //////*******************************************************************************

                    ////printErrorsOfTree(NwaveletsRF, analysisFolderName + "\\Num_of_RF_wavelets_validationDB.txt");

                    ////printErrorsOfTree(missLabelsRF, NwaveletsRF, analysisFolderName + "\\RF_MissLabeling_by_wavelets.txt");                
                }

                if (Form1.runRFProoning)
                {
                    int topLevel = int.MaxValue;
                    for (int k = 0; k < RFdecTreeArr.Count(); k++)
                    {
                        int tmp = getTopLevel(RFdecTreeArr[k]);
                        if (tmp < topLevel)
                            topLevel = tmp;
                    }

                    int topLevelBeginRF = rc.RFpruningTestRange[0];
                    topLevel = rc.RFpruningTestRange[1] == 0 ? topLevel : rc.RFpruningTestRange[1];

                    double[] errorRFProoning = new double[topLevel];
                    double[] NwaveletsRFProoning = new double[topLevel];
                    double[][] errorManyRFProoning = new double[topLevel][];
                    double[][] errorManyRFProoningNoVoting = new double[topLevel][];
                    for (int i = 0; i < topLevel; i++)
                    {
                        errorManyRFProoning[i] = new double[RFdecTreeArr.Count()];
                        errorManyRFProoningNoVoting[i] = new double[RFdecTreeArr.Count()];
                    }


                    if (Form1.rumPrallel)
                    {
                        Parallel.For(0, topLevel, i =>
                        {
                            //errorRFProoning[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, 2);
                            errorManyRFProoning[i] = testDecisionTreeManyRF(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, rc.NormLPType);
                            if (Form1.u_config.estimateRFnoVotingCB == "1")
                                errorManyRFProoningNoVoting[i] = testDecisionTreeManyRFNoVoting(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, rc.NormLPType);
                            NwaveletsRFProoning[i] = i + 1;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                        });
                    }
                    else
                    {
                        for (int i = 0; i < topLevel; i++)
                        {
                            //errorRFProoning[i] = testDecisionTreeRF(db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, 2);
                            errorManyRFProoning[i] = testDecisionTreeManyRF(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, rc.NormLPType);
                            if (Form1.u_config.estimateRFnoVotingCB == "1")
                                errorManyRFProoningNoVoting[i] = testDecisionTreeManyRFNoVoting(testingArr, db.PCAvalidation_dt, db.validation_label, RFdecTreeArr, i + 1, rc.NormLPType);
                            NwaveletsRFProoning[i] = i + 1;//  / rc.rfNum - if we want to devide by the number of trees to get the degree 
                        }
                    }

                    int minErrPruningRF_index = Enumerable.Range(0, errorRFProoning.Length).Aggregate((a, b) => (errorRFProoning[a] < errorRFProoning[b]) ? a : b); //minerror
                    double lowest_TreePruningRF_error = testDecisionTreeRF(testingArr, db.PCAtesting_dt, db.testing_label, RFdecTreeArr, minErrPruningRF_index + 1, 2);
                    printErrorsOfTree(lowest_TreePruningRF_error, minErrPruningRF_index * rc.hopping_size, analysisFolderName + "\\bsp_tree_errors_by_Pruning_RF_TestDB.txt");

                    //printErrorsOfTree(errorRFProoning, NwaveletsRFProoning, analysisFolderName + "\\RF_errors_by_Pruning.txt");
                    Form1.printtable(errorManyRFProoning, analysisFolderName + "\\cumulative_RF_errors_by_Pruning_validationDB.txt");

                    if (Form1.u_config.estimateRFnoVotingCB == "1")
                        Form1.printtable(errorManyRFProoningNoVoting, analysisFolderName + "\\independent_errors_of_rf_treees_no_voting_Pruning_validationDB.txt");
                    //printErrorsOfTree(NwaveletsRFProoning, analysisFolderName + "\\Num_of_ManyRF_levels_validationDB.txt");              
                }
            }

            #endregion

            #region Boosting tree

            if (Form1.runBoosting)
            {
                //BOOST
                List<GeoWave>[] BoostTreeArr = new List<GeoWave>[rc.boostNum]; //
                double[][] boostedLabels = new double[db.training_label.Count()][];
                // this for loop 
                for (int i = 0; i < db.training_label.Count(); i++)
                {
                    boostedLabels[i] = new double[db.training_label[0].Count()];
                    for (int j = 0; j < db.training_label[0].Count(); j++)
                        boostedLabels[i][j] = db.training_label[i][j];
                }

                bool[] Dim2Take = getDim2Take(rc, 0);//should take all


                //Array.Copy(db.training_label, 0, boostedLabels, 0, db.training_label.Length); - bad copy - by reference
                double[] best_norms = new double[rc.boostNum];
                int[] best_indeces = new int[rc.boostNum];
                for (int i = 0; i < rc.boostNum; i++)
                {
                    decicionTree decTreeBoost = new decicionTree(rc, db.PCAtraining_dt, boostedLabels, db.PCAtraining_GridIndex_dt, Dim2Take);
                    if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArr[i] = decision_GeoWaveArr;//take tree from first creation of "BSP" tree
                    else
                        BoostTreeArr[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);


                    //KFUNC
                    //best_indeces[i] = getGWIndexByKfunc(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i]);
                    best_indeces[i] = getGWIndexByKfuncLessAcurate(BoostTreeArr[i], rc, db.PCAtraining_dt, boostedLabels, ref best_norms[i], testingArr);
                    best_norms[i] = BoostTreeArr[i][best_indeces[i]].norm;

                    boostedLabels = GetResidualLabelsInBoosting(BoostTreeArr[i], db.PCAtraining_dt, boostedLabels, best_norms[i]);
                    //Form1.printtable(boostedLabels, analysisFolderName + "\\BoostingLabels_" + i.ToString() + "_tree.txt");
                    rc.boostlamda_0 = rc.boostlamda_0 * 0.5;

                    //dbg
                    //Form1.printConstWavelets2File(BoostTreeArr[i], analysisFolderName + "\\BoostingdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                }

                double[] tmpArr = new double[BoostTreeArr.Count()];
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                {
                    tmpArr[i] = Convert.ToDouble(best_indeces[i]);
                }
                printErrorsOfTree(tmpArr, analysisFolderName + "\\num_wavelets_in_boosting.txt");
                printErrorsOfTree(best_norms, analysisFolderName + "\\threshold_norms_of_wavelets_in_boosting.txt");

                //TEST IT
                List<double> NormArrBoosting = new List<double>();
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                    for (int j = 0; j < BoostTreeArr[i].Count; j++)
                        if (BoostTreeArr[i][j].norm >= best_norms[i])
                            NormArrBoosting.Add(BoostTreeArr[i][j].norm);
                NormArrBoosting = NormArrBoosting.OrderByDescending(o => o).ToList();

                int arrSizeBoost = Convert.ToInt32(rc.test_error_size * NormArrBoosting.Count / rc.hopping_size);
                double[] errorBoosting = new double[arrSizeBoost];
                //double[] missLabelsBoosting = new double[arrSizeBoost];
                double[] missLabelsBoostingBER = new double[arrSizeBoost];
                double[] NwaveletsBoosting = new double[arrSizeBoost];

                if (Form1.rumPrallel)
                {
                    Parallel.For(0, arrSizeBoost, i =>
                    {
                        errorBoosting[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArrBoosting[i * rc.hopping_size], 2, best_norms);
                        NwaveletsBoosting[i] = i * rc.hopping_size;
                    });
                }
                else
                {
                    for (int i = 0; i < arrSizeBoost; i++)
                    {
                        errorBoosting[i] = testDecisionTreeBoosting(db.PCAtesting_dt, db.testing_label, BoostTreeArr, NormArrBoosting[i * rc.hopping_size], 2, best_norms);
                        NwaveletsBoosting[i] = i * rc.hopping_size;
                    }
                }
                printErrorsOfTree(errorBoosting, NwaveletsBoosting, analysisFolderName + "\\Boosting_errors_by_wavelets.txt");
            }



            #endregion

            #region Prooning Boosting tree

            if (Form1.runBoostingProoning)// Gradient Boosting
            {
                List<GeoWave>[] BoostTreeArrPooning = new List<GeoWave>[rc.boostNum]; // מערך המכיל את כל העצים הנבנים לאורך האיטרציות

                //  1) Initialize f0(x) = argmin_γ sum L(yi, γ).
                //  commemt: L(x,y)=(x-y)^2 => f0(x) = mean(yi)

                double[] DBmeanVal = new double[trainingArr.Count()]; // חישוב הערך הממוצע בכל סט האימון לטובת ניחוש ראשוני
                //  initialization
                for (int i = 0; i < trainingArr.Count(); i++) // לולאה על מספר הלייבלים בסט אימון
                    for (int j = 0; j < db.training_label[0].Count(); j++)
                        DBmeanVal[j] += db.training_label[trainingArr[i]][j];
                //  calc mean val
                for (int j = 0; j < db.training_label[0].Count(); j++)
                    DBmeanVal[j] = DBmeanVal[j] / trainingArr.Count();

                double[][] F = new double[rc.boostNum + 1][];

                for (int i = 0; i < F.Count(); i++)
                    F[i] = new double[db.training_label.Count()];
                // f0(x) = mean(yi)
                for (int PointInd = 0; PointInd < db.training_label.Count(); PointInd++) // לולאה על מספר הלייבלים בסט אימון
                    F[0][PointInd] = DBmeanVal[0];

                // 2) For i = 1 2 ... M:

                bool[] Dim2Take = getDim2Take(rc, 0); //should take all
                int[] best_level = new int[rc.boostNum]; // מערך בגודל מספר איטרציות - אמור להכיל את ה"רמה" הטובה ביותר שנבחר על פיה לבצע את האיטרציה - במקרה הזה הרמה קבועה לכל האיטרציות 
                int[] best_indecesProoning = new int[rc.boostNum]; // לא מצאתי משמעות
                double[][] boostedLabelsPooning = new double[db.training_label.Count()][]; // מערך בגודל סט האימון - מכיל את הלייבלים המוערכים ע"י האלגוריתם

                List<int> GBoriginalTrainArr = trainingArr;
                double percent_of_bagging_on_GB = bag_fraction;
                // * TODO: fix SGBM just like i fixed WGBM
                for (int i = 0; i < rc.boostNum; i++)
                {
                    // 2(a) for each of the training points i (m is the iteration number)
                    // r_[i][m] = y[i] - f_m[i]; (this is the gradient of the cost function L(x,y)=(x-y)^2
                    // r - boostedLabelsPooning, y - db.training_label, f - F

                    if (percent_of_bagging_on_GB < 1) // Stochastic GB
                    {
                        GBseed++;
                        var ran = new Random(GBseed);
                        int OOBlen = Convert.ToUInt16(Math.Floor((1 - percent_of_bagging_on_GB) * GBoriginalTrainArr.Count()));

                        trainingArr = GBoriginalTrainArr.OrderBy(x => ran.Next()).ToList().GetRange(OOBlen, GBoriginalTrainArr.Count() - OOBlen);
                    }

                    if (Form1.rumPrallel)
                    {
                        Parallel.For(0, trainingArr.Count(), PointInd =>
                        {
                            boostedLabelsPooning[trainingArr[PointInd]] = new double[db.training_label[0].Count()]; // מגדיר כל "איבר" במערך הלייבלים - לכל לייבל משתנה מסוג דאבל..
                            for (int j = 0; j < db.training_label[0].Count(); j++) // לולאה על מספר הלייבלים בכל שורה (במידה ויש יותר מלייבל אחד לכל אובייט).
                                boostedLabelsPooning[trainingArr[PointInd]][j] = db.training_label[trainingArr[PointInd]][j] - F[i][trainingArr[PointInd]]; // כאן אורן הכניס לרנינג ריט. לפי הספר של פרידמן צריך להכניס רק בסוף בחישוב הפונקציה
                        });
                    }
                    else
                    {
                        for (int PointInd = 0; PointInd < trainingArr.Count(); PointInd++) // לולאה על מספר הלייבלים בסט אימון
                        {
                            boostedLabelsPooning[trainingArr[PointInd]] = new double[db.training_label[0].Count()]; // מגדיר כל "איבר" במערך הלייבלים - לכל לייבל משתנה מסוג דאבל..
                            for (int j = 0; j < db.training_label[0].Count(); j++) // לולאה על מספר הלייבלים בכל שורה (במידה ויש יותר מלייבל אחד לכל אובייט).
                                boostedLabelsPooning[trainingArr[PointInd]][j] = db.training_label[trainingArr[PointInd]][j] - F[i][trainingArr[PointInd]]; // כאן אורן הכניס לרנינג ריט. לפי הספר של פרידמן צריך להכניס רק בסוף בחישוב הפונקציה
                        }
                    }

                    double[] boostedLabelsPooningToPrint = new double[boostedLabelsPooning.Count()];
                    for (int ind = 0; ind < boostedLabelsPooning.Count(); ind++)
                        if (boostedLabelsPooning[ind] == null)
                            boostedLabelsPooningToPrint[ind] = 0;
                        else
                            boostedLabelsPooningToPrint[ind] = boostedLabelsPooning[ind][0];

                    int printRES = 0;

                    if (printRES == 1)
                        printErrorsOfTree(boostedLabelsPooningToPrint, analysisFolderName + "\\residual" + (i + 1).ToString() + ".txt");

                    // (b) Fit a regression tree to the targets r_im giving terminal regions R_jm, j = 1, 2, . . . , Jm. Jm is the "best level" - my choise.

                    decicionTree decTreeBoost = new decicionTree(rc, db.PCAtraining_dt, boostedLabelsPooning, db.PCAtraining_GridIndex_dt, Dim2Take);//  הגדרת עץ חדש - רק מבנה נתונים ולא ממש עץ עם עלים 

                    /*if (i == 0 && decision_GeoWaveArr.Count > 0)
                        BoostTreeArrPooning[i] = decision_GeoWaveArr; //take tree from first creation of "BSPee
                    else
                    */
                    BoostTreeArrPooning[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);

                    best_level[i] = Convert.ToInt32(rc.boostProoning_0);

                    // (c) For j = 1, 2, . . . , Jm compute
                    // γjm = argmin_γ( sum_xi∈Rjm  L(yi, f_m−1(xi) + γ) . 

                    // (d) fm(x) = fm−1(x) + LR * SUM(γ_jm*I(x ∈ Rjm))
                    BoostTreeArrPooning[i] = BoostTreeArrPooning[i].OrderBy(o => o.ID).ToList();

                    for (int PointInd = 0; PointInd < db.training_dt.Count(); PointInd++)
                        F[i + 1][PointInd] = F[i][PointInd] + rc.learningRate * askTreeMeanValAtLevel(db.training_dt[PointInd], BoostTreeArrPooning[i], best_level[i])[0];

                    int printF = 0;

                    if (printF == 1)
                    {
                        printErrorsOfTree(F[i + 1], analysisFolderName + "\\F" + (i + 1).ToString() + ".txt");
                        printErrorsOfTree(F[i], analysisFolderName + "\\F" + (i).ToString() + ".txt");

                    }

                }
                // Print error and others...
                trainingArr = GBoriginalTrainArr;
                //printErrorsOfTree(F[rc.boostNum], analysisFolderName + "\\F" + (rc.boostNum).ToString() + ".txt");
                double[] TrainARRforPrint = new double[trainingArr.Count() + validationArr.Count()];
                for (int ind = 0; ind < trainingArr.Count(); ind++)
                    TrainARRforPrint[ind] = trainingArr[ind];
                for (int ind = 0; ind < validationArr.Count(); ind++)
                    TrainARRforPrint[trainingArr.Count() + ind] = validationArr[ind];
                printErrorsOfTree(TrainARRforPrint, analysisFolderName + "\\TrainARR.txt");

                printErrorsOfTree(TrainARRforPrint, analysisFolderName + "\\TrainARR.txt");


                double[] tmpArr = new double[BoostTreeArrPooning.Count() + 1];
                tmpArr[0] = 0;
                for (int i = 1; i < BoostTreeArrPooning.Count() + 1; i++)
                {
                    tmpArr[i] = Convert.ToDouble(best_level[i - 1]);
                }
                printErrorsOfTree(tmpArr, analysisFolderName + "\\tree_levels_in_boosting.txt");

                double[] errorOnTest = new double[rc.boostNum + 1];
                double[] errorOnTrain = new double[rc.boostNum + 1];

                errorOnTest = computeErrorOnTest(F, testingArr, trainingArr, rc.boostNum, LPnorm2Error);
                printErrorsOfTree(errorOnTest, tmpArr, analysisFolderName + "\\Boosting_Prooning_errors_by_levels.txt");

                //errorOnTest = computeErrorOnThisArr(F, trainingArr, trainingArr, rc.boostNum, LPnorm2Error);
                //printErrorsOfTree(errorOnTest, tmpArr, analysisFolderName + "\\Boosting_Prooning_errors_by_levels.txt");


                errorOnTrain = computeErrorOnThisArr(F, trainingArr, trainingArr, rc.boostNum, LPnorm2Error);
                printErrorsOfTree(errorOnTrain, tmpArr, analysisFolderName + "\\Boosting_Prooning_errors_by_levels_ON_TRAIN.txt");

                if (rc.percent_training_db < 1)
                {
                    double[] errorOnValid = new double[rc.boostNumLearningRate + 1];
                    errorOnValid = computeErrorOnThisArr(F, validationArr, trainingArr, rc.boostNum, LPnorm2Error);
                    printErrorsOfTree(errorOnValid, tmpArr, analysisFolderName + "\\Boosting_Prooning_errors_by_levels_ON_VALID.txt");
                }


                if (Form1.u_config.saveTressCB == "1")
                {
                    if (!System.IO.Directory.Exists(analysisFolderName + "\\archive"))
                        System.IO.Directory.CreateDirectory(analysisFolderName + "\\archive");
                    //Form1.printtable(trainingArrRF_indecesList, analysisFolderName + "\\RFIndeces.txt"); //- not for giant DB
                    for (int i = 0; i < BoostTreeArrPooning.Count(); i++)
                    {
                        Form1.printWaveletsProperties(BoostTreeArrPooning[i], analysisFolderName + "\\archive\\waveletsPropertiesTree_" + i.ToString() + ".txt");
                        //Form1.printConstWavelets2File(RFdecTreeArr[i], analysisFolderName + "\\archive\\RFdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                    }
                }
            }

            #endregion

            #region Boosting tree LearningRate

            if (Form1.runBoostingLearningRate) // WGBM
            {
                List<GeoWave>[] BoostTreeArrLearningRate = new List<GeoWave>[rc.boostNumLearningRate]; // מערך המכיל את כל העצים הנבנים לאוך האיטרציות

                //  1) Initialize F0(x) = argmin_γ sum L(yi, γ).
                //  L(x,y)=(x-y)^2 => F0(x) = mean(yi)
                //  F[i] is the algorithm approximation at iteration i 

                double[] DBmeanVal = new double[trainingArr.Count()]; // חישוב הערך הממוצע בכל סט האימון לטובת ניחוש ראשוני
                for (int i = 0; i < trainingArr.Count(); i++) // לולאה על מספר הלייבלים בסט אימון
                    for (int j = 0; j < db.training_label[0].Count(); j++)
                        DBmeanVal[j] += db.training_label[trainingArr[i]][j];
                for (int j = 0; j < db.training_label[0].Count(); j++)
                    DBmeanVal[j] = DBmeanVal[j] / trainingArr.Count();

                double[][] F = new double[rc.boostNumLearningRate + 1][];
                for (int i = 0; i < F.Count(); i++)
                    F[i] = new double[db.training_label.Count()];
                for (int PointInd = 0; PointInd < db.training_label.Count(); PointInd++)
                    F[0][PointInd] = DBmeanVal[0];


                // 2) For k = 1 2 ... K:
                // K is the number of iteration

                double[] BoostArrLearningRateNorms = new double[rc.boostNumLearningRate];
                double[][] boostedLabelsLearningRate = new double[db.training_label.Count()][];

                bool[] Dim2Take = getDim2Take(rc, 0);

                double[] NumOfLeaves = new double[rc.boostNumLearningRate + 1];
                double[] lambda_arr = new double[rc.boostNumLearningRate];
                double lambda = 1;
                NumOfLeaves[0] = 0;
                List<int> WGBoriginalTrainArr = trainingArr;
                double percent_of_bagging_on_WGB = bag_fraction;

                for (int i = 0; i < rc.boostNumLearningRate; i++)
                {
                    List<int> WGBvalidationArr = new List<int>();
                    if (percent_of_bagging_on_WGB < 1)
                    {
                        WGBseed++;
                        var ran = new Random(WGBseed);
                        int validLength = Convert.ToUInt16(Math.Floor((1 - percent_of_bagging_on_WGB) * WGBoriginalTrainArr.Count()));
                        List<int> RandOrderdTrain = WGBoriginalTrainArr.OrderBy(x => ran.Next()).ToList();

                        trainingArr = RandOrderdTrain.GetRange(validLength, WGBoriginalTrainArr.Count() - validLength);
                        WGBvalidationArr = RandOrderdTrain.GetRange(0, validLength);
                        
                    }


                    // 2(a) for each of the training points (i - Point, k - Iteration)
                    // r_[i][k] = y[i] - f_m[i];
                    // r - boostedLabelsLearningRate, y - db.training_label, f - F

                    //  compute the residual for each point in the data base (or the gradient of the residual depends on L)

                    for (int PointInd = 0; PointInd < WGBoriginalTrainArr.Count(); PointInd++)
                    {
                        boostedLabelsLearningRate[WGBoriginalTrainArr[PointInd]] = new double[db.training_label[0].Count()];
                        for (int j = 0; j < db.training_label[0].Count(); j++)
                            boostedLabelsLearningRate[WGBoriginalTrainArr[PointInd]][j] = db.training_label[WGBoriginalTrainArr[PointInd]][j] - F[i][WGBoriginalTrainArr[PointInd]];
                    }

                    int printRES = 0;
                    if (printRES == 1)
                    {
                        double[] boostedLabelsLearningRateToPrint = new double[boostedLabelsLearningRate.Count()];
                        for (int ind = 0; ind < boostedLabelsLearningRate.Count(); ind++)
                            if (boostedLabelsLearningRate[ind] == null)
                                boostedLabelsLearningRateToPrint[ind] = 0;
                            else
                                boostedLabelsLearningRateToPrint[ind] = boostedLabelsLearningRate[ind][0];
                        printErrorsOfTree(boostedLabelsLearningRateToPrint, analysisFolderName + "\\residual" + (i + 1).ToString() + ".txt");
                    }

                    // (b) Fit a regression tree to the targets r_im giving terminal regions R_jm, j = 1, 2, . . . , Jm. Jm is the "best level".

                    decicionTree decTreeBoost = new decicionTree(rc, db.PCAtraining_dt, boostedLabelsLearningRate, db.PCAtraining_GridIndex_dt, Dim2Take);

                    if (i == -1)
                        BoostTreeArrLearningRate[i] = decision_GeoWaveArr;
                    else
                        BoostTreeArrLearningRate[i] = decTreeBoost.getdecicionTree(trainingArr, boundingBox);
                    // ALTOUGH THE LABLE VECTOR IS "FULL" WE USE ONLY TRAININGARR TO CONSTRUCT THE DATA

                    // Compute best number of wavelets to take

                    BoostArrLearningRateNorms[i] = GetWaveletNormThByWGBvalidation(BoostTreeArrLearningRate[i], WGBvalidationArr, db.PCAtraining_dt,
                            boostedLabelsLearningRate, i, analysisFolderName, lambda);
                    lambda_arr[i] = lambda;
                    NumOfLeaves[i + 1] = GetNumOfLeafsByNormTH(BoostTreeArrLearningRate[i], BoostArrLearningRateNorms[i]);
                    
                    // (c) For j = 1, 2, . . . , Jm compute
                    // γjm = argmin_γ( sum_xi∈Rjm  L(yi, f_m−1(xi) + γ) . 

                    // (d) fm(x) = fm−1(x) + LR * SUM(γ_jm*I(x ∈ Rjm))

                    BoostTreeArrLearningRate[i] = BoostTreeArrLearningRate[i].OrderBy(o => o.ID).ToList();
                    for (int PointInd = 0; PointInd < db.training_label.Count(); PointInd++)
                        F[i + 1][PointInd] = F[i][PointInd] + rc.learningRate * askTreeMeanVal(db.training_dt[PointInd], BoostTreeArrLearningRate[i], BoostArrLearningRateNorms[i])[0];
                    BoostTreeArrLearningRate[i] = BoostTreeArrLearningRate[i].OrderBy(o => o.norm).ToList();

                    //  db.training_label
                    int printF = 0;

                    if (printF == 1)
                        printErrorsOfTree(F[i + 1], analysisFolderName + "\\F" + (i + 1).ToString() + ".txt");

                }

                trainingArr = WGBoriginalTrainArr;

                // Print error and others...

                double[] TrainARRforPrint = new double[trainingArr.Count() + validationArr.Count()];
                for (int ind = 0; ind < trainingArr.Count(); ind++)
                    TrainARRforPrint[ind] = trainingArr[ind];
                for (int ind = 0; ind < validationArr.Count(); ind++)
                    TrainARRforPrint[trainingArr.Count() + ind] = validationArr[ind];
                printErrorsOfTree(TrainARRforPrint, analysisFolderName + "\\TrainARR.txt");

                double[] errorOnTest = new double[rc.boostNumLearningRate + 1];
                double[] errorOnTrain = new double[rc.boostNumLearningRate + 1];

                //TODO: SEND THIS PARAMETERS FROM GUI.
                Boolean useAUC = false;

                if (useAUC == false)
                {
                    errorOnTest = computeErrorOnTest(F, testingArr, WGBoriginalTrainArr, rc.boostNumLearningRate, LPnorm2Error);
                    printErrorsOfTree(errorOnTest, NumOfLeaves, analysisFolderName + "\\BoostTreeArrLearningRateError.txt");
                }
                else
                {
                    errorOnTest = computeAUCOnTest(F, testingArr, WGBoriginalTrainArr, rc.boostNumLearningRate, LPnorm2Error);
                    printErrorsOfTree(errorOnTest, NumOfLeaves, analysisFolderName + "\\BoostTreeArrLearningRateAUC.txt");

                }

                errorOnTrain = computeErrorOnTrain(F, testingArr, WGBoriginalTrainArr, rc.boostNumLearningRate, LPnorm2Error);
                printErrorsOfTree(errorOnTrain, NumOfLeaves, analysisFolderName + "\\BoostTreeArrLearningRateError_ON_TRAIN.txt");

                if (rc.percent_training_db < 1)
                {
                    double[] errorOnValid = new double[rc.boostNumLearningRate + 1];
                    if (useAUC == false)
                    {
                        errorOnValid = computeErrorOnTest(F, validationArr, WGBoriginalTrainArr, rc.boostNumLearningRate, LPnorm2Error);
                        printErrorsOfTree(errorOnValid, NumOfLeaves, analysisFolderName + "\\BoostTreeArrLearningRateError_ON_VALID.txt");
                    }
                    else
                    {
                        errorOnValid = computeAUCOnTest(F, validationArr, WGBoriginalTrainArr, rc.boostNumLearningRate, LPnorm2Error);
                        printErrorsOfTree(errorOnValid, NumOfLeaves, analysisFolderName + "\\BoostTreeArrLearningRateAUC_ON_VALID.txt");
                    }
                }

                if (Form1.u_config.saveTressCB == "1")
                {
                    if (!System.IO.Directory.Exists(analysisFolderName + "\\archive"))
                        System.IO.Directory.CreateDirectory(analysisFolderName + "\\archive");
                    //Form1.printtable(trainingArrRF_indecesList, analysisFolderName + "\\RFIndeces.txt"); //- not for giant DB
                    for (int i = 0; i < BoostTreeArrLearningRate.Count(); i++)
                    {
                        Form1.printWaveletsProperties(BoostTreeArrLearningRate[i], analysisFolderName + "\\archive\\waveletsPropertiesTree_" + i.ToString() + ".txt");
                        //Form1.printConstWavelets2File(RFdecTreeArr[i], analysisFolderName + "\\archive\\RFdecTreeArr_" + i.ToString() + "_tree.txt");//dbg
                    }
                }

            }
            #endregion
        }

        private double[] computeErrorOnThisArr(double[][] F, List<int> ThisArr, List<int> trainingArr, int NumOfIter, int LPnorm2Error)
        {
            double lableAvg = 0;
            for (int ind = 0; ind < trainingArr.Count(); ind++)
                lableAvg += db.training_label[trainingArr[ind]][0];
            lableAvg = lableAvg / trainingArr.Count();

            int zeroGuess = 0;
            if (lableAvg <= 0)
                zeroGuess = -1;
            else
                zeroGuess = 1;

            double[] errorOnThis = new double[NumOfIter + 1];

            if (LPnorm2Error == 2)
                if (Form1.rumPrallel)
                {
                    Parallel.For(0, NumOfIter + 1, iter =>
                    {
                        for (int PointInd = 0; PointInd < ThisArr.Count(); PointInd++)
                            errorOnThis[iter] += Math.Pow(F[iter][ThisArr[PointInd]] - db.training_label[ThisArr[PointInd]][0], 2);
                        errorOnThis[iter] = Math.Pow(errorOnThis[iter] / ThisArr.Count(), 0.5);

                    });
                }
                else
                {
                    for (int iter = 0; iter < NumOfIter + 1; iter++)
                    {
                        for (int PointInd = 0; PointInd < ThisArr.Count(); PointInd++)
                            errorOnThis[iter] += Math.Pow(F[iter][ThisArr[PointInd]] - db.training_label[ThisArr[PointInd]][0], 2);
                        errorOnThis[iter] = Math.Pow(errorOnThis[iter] / ThisArr.Count(), 0.5);
                    }
                }
            else //classification +-1
            if (Form1.rumPrallel)
            {
                Parallel.For(0, NumOfIter + 1, iter =>
                {
                    for (int PointInd = 0; PointInd < ThisArr.Count(); PointInd++)
                    {
                        if (F[iter][ThisArr[PointInd]] * db.training_label[ThisArr[PointInd]][0] < 0) //train error rate 
            errorOnThis[iter] += 1;

                        if (F[iter][ThisArr[PointInd]] == 0 && db.training_label[ThisArr[PointInd]][0] != zeroGuess)
                            errorOnThis[iter] += 1;
                    }
                    errorOnThis[iter] = errorOnThis[iter] / ThisArr.Count();
                });
            }
            else
            {
                for (int iter = 0; iter < NumOfIter + 1; iter++)
                {
                    for (int PointInd = 0; PointInd < ThisArr.Count(); PointInd++)
                    {
                        if (F[iter][ThisArr[PointInd]] * db.training_label[ThisArr[PointInd]][0] < 0) //train error rate 
                            errorOnThis[iter] += 1;

                        if (F[iter][ThisArr[PointInd]] == 0 && db.training_label[ThisArr[PointInd]][0] != zeroGuess)
                            errorOnThis[iter] += 1;
                    }
                    errorOnThis[iter] = errorOnThis[iter] / ThisArr.Count();
                }
            }

            return errorOnThis;
        }

        private double[] computeErrorOnTrain(double[][] F, List<int> testingArr, List<int> trainingArr, int NumOfIter, int LPnorm2Error)
        {
            double lableAvg = 0;
            for (int ind = 0; ind < trainingArr.Count(); ind++)
                lableAvg += db.training_label[trainingArr[ind]][0];
            lableAvg = lableAvg / trainingArr.Count();

            int zeroGuess = 0;
            if (lableAvg <= 0)
                zeroGuess = -1;
            else
                zeroGuess = 1;

            double[] errorOnTrain = new double[NumOfIter + 1];

            if (LPnorm2Error == 2)
                if (Form1.rumPrallel)
                {
                    Parallel.For(0, NumOfIter + 1, iter =>
                    {
                        for (int PointInd = 0; PointInd < trainingArr.Count(); PointInd++)
                            errorOnTrain[iter] += Math.Pow(F[iter][trainingArr[PointInd]] - db.training_label[trainingArr[PointInd]][0], 2);
                        errorOnTrain[iter] = Math.Pow(errorOnTrain[iter] / trainingArr.Count(), 0.5);

                    });
                }
                else
                {
                    for (int iter = 0; iter < NumOfIter + 1; iter++)
                    {
                        for (int PointInd = 0; PointInd < trainingArr.Count(); PointInd++)
                            errorOnTrain[iter] += Math.Pow(F[iter][trainingArr[PointInd]] - db.training_label[trainingArr[PointInd]][0], 2);
                        errorOnTrain[iter] = Math.Pow(errorOnTrain[iter] / trainingArr.Count(), 0.5);
                    }
                }
            else //classification +-1
            if (Form1.rumPrallel)
            {
                Parallel.For(0, NumOfIter + 1, iter =>
                {
                    for (int PointInd = 0; PointInd < trainingArr.Count(); PointInd++)
                    {
                        if (F[iter][trainingArr[PointInd]] * db.training_label[trainingArr[PointInd]][0] < 0) //train error rate 
            errorOnTrain[iter] += 1;

                        if (F[iter][trainingArr[PointInd]] == 0 && db.training_label[trainingArr[PointInd]][0] != zeroGuess)
                            errorOnTrain[iter] += 1;
                    }
                    errorOnTrain[iter] = errorOnTrain[iter] / trainingArr.Count();
                });
            }
            else
            {
                for (int iter = 0; iter < NumOfIter + 1; iter++)
                {
                    for (int PointInd = 0; PointInd < trainingArr.Count(); PointInd++)
                    {
                        if (F[iter][trainingArr[PointInd]] * db.training_label[trainingArr[PointInd]][0] < 0) //train error rate 
                            errorOnTrain[iter] += 1;

                        if (F[iter][trainingArr[PointInd]] == 0 && db.training_label[trainingArr[PointInd]][0] != zeroGuess)
                            errorOnTrain[iter] += 1;
                    }
                    errorOnTrain[iter] = errorOnTrain[iter] / trainingArr.Count();
                }
            }

            return errorOnTrain;
        }

        private double[] computeErrorOnTest(double[][] F, List<int> testingArr, List<int> trainingArr, int NumOfIter, int LPnorm2Error)
        {
            double lableAvg = 0;
            for (int ind = 0; ind < trainingArr.Count(); ind++)
                lableAvg += db.training_label[trainingArr[ind]][0];
            lableAvg = lableAvg / trainingArr.Count();

            int zeroGuess = 0;
            if (lableAvg <= 0)
                zeroGuess = -1;
            else
                zeroGuess = 1;

            double[] errorOnTest = new double[NumOfIter + 1];
            if (LPnorm2Error == 2)
                if (Form1.rumPrallel)
                {
                    Parallel.For(0, NumOfIter + 1, iter =>
                    {
                        for (int PointInd = 0; PointInd < testingArr.Count(); PointInd++)
                            errorOnTest[iter] += Math.Pow(F[iter][testingArr[PointInd]] - db.testing_label[testingArr[PointInd]][0], 2);
                        errorOnTest[iter] = Math.Pow(errorOnTest[iter] / testingArr.Count(), 0.5);

                    });
                }
                else
                {
                    for (int iter = 0; iter < NumOfIter + 1; iter++)
                    {
                        for (int PointInd = 0; PointInd < testingArr.Count(); PointInd++)
                            errorOnTest[iter] += Math.Pow(F[iter][testingArr[PointInd]] - db.testing_label[testingArr[PointInd]][0], 2);
                        errorOnTest[iter] = Math.Pow(errorOnTest[iter] / testingArr.Count(), 0.5);
                    }
                }
            else  //classification +-1
            if (Form1.rumPrallel)
            {
                Parallel.For(0, NumOfIter + 1, iter =>
                {
                    for (int PointInd = 0; PointInd < testingArr.Count(); PointInd++)
                    {
                        if (F[iter][testingArr[PointInd]] * db.testing_label[testingArr[PointInd]][0] < 0) //test error rate 
            errorOnTest[iter] += 1;

                        if (F[iter][testingArr[PointInd]] == 0 && db.testing_label[testingArr[PointInd]][0] != zeroGuess) //test error rate 
            errorOnTest[iter] += 1;
                    }
                    errorOnTest[iter] = errorOnTest[iter] / testingArr.Count();
                });
            }
            else
            {
                for (int iter = 0; iter < NumOfIter + 1; iter++)
                {
                    for (int PointInd = 0; PointInd < testingArr.Count(); PointInd++)
                    {
                        if (F[iter][testingArr[PointInd]] * db.testing_label[testingArr[PointInd]][0] < 0) //test error rate 
                            errorOnTest[iter] += 1;

                        if (F[iter][testingArr[PointInd]] == 0 && db.testing_label[testingArr[PointInd]][0] != zeroGuess) //test error rate 
                            errorOnTest[iter] += 1;
                    }
                    errorOnTest[iter] = errorOnTest[iter] / testingArr.Count();
                }
            }
            return errorOnTest;
        }
        private double[] computeAUCOnTest(double[][] F, List<int> testingArr, List<int> trainingArr, int NumOfIter, int LPnorm2Error)
        {
            double[] AUCOnTest = new double[NumOfIter + 1];
            double[][] RocCurveFPR = new double[NumOfIter + 1][];
            double[][] RocCurveTPR = new double[NumOfIter + 1][];
            double[] AUC = new double[NumOfIter + 1];
            int AUC_inc_res = 100;

            for (int i = 0; i < RocCurveFPR.Count(); i++)
            {
                RocCurveFPR[i] = new double[AUC_inc_res + 1];
                RocCurveTPR[i] = new double[AUC_inc_res + 1];
            }
            for (int i = 0; i <= AUC_inc_res; i++)
            {
                double[][] F_tmp = new double[F.Count()][];

                for (int k = 0; k < F_tmp.Count(); k++)
                {
                    F_tmp[k] = new double[F[0].Count()];
                    for (int j = 0; j < F_tmp[k].Count(); j++)
                        F_tmp[k][j] = F[k][j];
                }

                double th = -1 + 0.02 * i;
                for (int j = 0; j < F_tmp.Count(); j++)
                {
                    for (int k = 0; k < F_tmp[j].Count(); k++)
                    {
                        if (F_tmp[j][k] <= th)
                            F_tmp[j][k] = -1;
                        else
                            F_tmp[j][k] = 1;
                    }
                }


                for (int iter = 0; iter < NumOfIter + 1; iter++)
                {
                    double TrueP = 0;
                    double FalseP = 0;
                    double TrueN = 0;
                    double FalseN = 0;

                    for (int PointInd = 0; PointInd < testingArr.Count(); PointInd++)
                    {
                        if (db.testing_label[testingArr[PointInd]][0] == 1)
                            if (F_tmp[iter][testingArr[PointInd]] == 1)
                                TrueN = TrueN + 1;
                            else
                                FalseP = FalseP + 1;
                        if (db.testing_label[testingArr[PointInd]][0] == -1)
                            if (F_tmp[iter][testingArr[PointInd]] == -1)
                                TrueP = TrueP + 1;
                            else
                                FalseN = FalseN + 1;
                    }
                    if (FalseP + TrueN == 0)
                        RocCurveFPR[iter][i] = 0;
                    else
                        RocCurveFPR[iter][i] = FalseP / (FalseP + TrueN);

                    if (TrueP + FalseN == 0)
                        RocCurveTPR[iter][i] = 0;
                    else
                        RocCurveTPR[iter][i] = TrueP / (TrueP + FalseN);
                }
            }

            for (int iter = 0; iter < AUC.Count(); iter++)
            {
                AUC[iter] = 0;
                for (int t = 0; t < RocCurveTPR[iter].Count() - 1; t++)
                {
                    double dx = RocCurveFPR[iter][t + 1] - RocCurveFPR[iter][t];
                    double dy = RocCurveTPR[iter][t + 1] - RocCurveTPR[iter][t];
                    AUC[iter] = AUC[iter] + dx * dy / 2 + dx * RocCurveTPR[iter][t];
                }

            }

            return AUC;
        }

        private double[][] computeTreeEstimatedLables(List<int> testingArr, double[][] testing_dt, List<GeoWave>[] BoostTreeArr,
        double maxNorm, int Ntrees, int lablesCount)
        {
            double[][] estimatedLabels = new double[testingArr.Count()][];

            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[lablesCount];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    double[] point = new double[rc.dim];
                    for (int j = 0; j < rc.dim; j++)
                        point[j] = double.Parse(testing_dt[testingArr[i]][j].ToString());

                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    for (int j = 0; j < Ntrees; j++)
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorm);

                    for (int j = 0; j < lablesCount; j++)
                        for (int k = 0; k < Ntrees; k++)
                            estimatedLabels[i][j] += (rc.learningRate * tmpLabel[k][j]);


                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    double[] point = new double[rc.dim];
                    for (int j = 0; j < rc.dim; j++)
                        point[j] = double.Parse(testing_dt[testingArr[i]][j].ToString());

                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    for (int j = 0; j < Ntrees; j++)
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorm);

                    for (int j = 0; j < lablesCount; j++)
                        for (int k = 0; k < Ntrees; k++)
                            estimatedLabels[i][j] += (rc.learningRate * tmpLabel[k][j]);

                }
            }


            return (estimatedLabels);
        }

    
        private double GetWaveletNormThByWGBvalidation(List<GeoWave> NormSortedTree, List<int> validationArr, double[][] TrainSet,
        double[][] Labels, int iteration, string analysisFolderName, double lambda)
        {

            int MaxNOW = Math.Min(9999999, NormSortedTree.Count());// now = number of wavelets
            List<GeoWave> Tree_orderedById = NormSortedTree.OrderBy(o => o.ID).ToList();
            double[] ErrorTable = new double[MaxNOW];
            double[] totalErrorTable = new double[MaxNOW];
            double[] Penalty = new double[MaxNOW];

            for (int ind = 0; ind < MaxNOW; ind++) //foundation for penalty
                Penalty[ind] = 0;
            if (Form1.rumPrallel)
            {
                Parallel.For(0, MaxNOW, WaveletIndex =>
                {
                    for (int pointInd = 0; pointInd < validationArr.Count(); pointInd++)
                    {
                        double[] point = new double[rc.dim];
                        for (int j = 0; j < rc.dim; j++)
                            point[j] = double.Parse(TrainSet[validationArr[pointInd]][j].ToString());

                        double[] currTreeLabel = askTreeMeanVal(point, Tree_orderedById, NormSortedTree[WaveletIndex].norm);

                        for (int j = 0; j < currTreeLabel.Count(); j++)
                            ErrorTable[WaveletIndex] += Math.Pow(Labels[validationArr[pointInd]][j] - currTreeLabel[j], 2);
                    }
                    totalErrorTable[WaveletIndex] = Math.Pow(ErrorTable[WaveletIndex], 0.5) + lambda * Penalty[WaveletIndex]; // lambda*||f-f^_N||^2_L2 + ||f^_N||_BV
                });
            }
            else
                for (int WaveletIndex = 0; WaveletIndex < MaxNOW; WaveletIndex++)
                {
                    for (int pointInd = 0; pointInd < validationArr.Count(); pointInd++)
                    {
                        double[] point = new double[rc.dim];
                        for (int j = 0; j < rc.dim; j++)
                            point[j] = double.Parse(TrainSet[validationArr[pointInd]][j].ToString());

                        double[] currTreeLabel = askTreeMeanVal(point, Tree_orderedById, NormSortedTree[WaveletIndex].norm);

                        for (int j = 0; j < currTreeLabel.Count(); j++)
                            ErrorTable[WaveletIndex] += Math.Pow(Labels[validationArr[pointInd]][j] - currTreeLabel[j], 2);
                    }
                    totalErrorTable[WaveletIndex] = Math.Pow(ErrorTable[WaveletIndex], 0.5) + lambda * Penalty[WaveletIndex]; // lambda*||f-f^_N||^2_L2 + ||f^_N||_BV
                }

            double totalMinVal = totalErrorTable[0];
            int MinInd = 0;
            for (int i = 1; i < totalErrorTable.Count(); i++)
            {
                if (totalErrorTable[i] < totalMinVal)
                {
                    MinInd = i;
                    totalMinVal = totalErrorTable[i];
                }

            }

            if (NormSortedTree.Count == 1)
                MinInd = 0;

            return NormSortedTree[MinInd].norm;
        }
        
        private List<int> getListFromFile(string fileName)
        {
            List<int> Arr = new List<int>();
            StreamReader sr = new StreamReader(File.OpenRead(fileName));

            while (!sr.EndOfStream)
            {
                Arr.Add(int.Parse(sr.ReadLine()));
            }
            sr.Close();
            return Arr;
        }

        private double GetThWaveletNorm(List<GeoWave> NormSortedTree)
        {

            double[] NormArr = new double[NormSortedTree.Count];
            double MaxNorm = NormSortedTree[0].norm;
            int i = 0;
            NormArr[0] = 100;
            do
            {
                i++;
                NormArr[i] = (NormSortedTree[i].norm * 100) / MaxNorm;
            } while (NormArr[i - 1] - NormArr[i] > 0.5 && i < NormSortedTree.Count - 1);

            return NormSortedTree[i].norm;
        }

        private double GetThWaveletNorm2(List<GeoWave> NormSortedTree)
        {

            double[] NormArr = new double[NormSortedTree.Count];
            double MaxNorm = NormSortedTree[0].norm;
            int i = 0;
            NormArr[0] = MaxNorm;
            do
            {
                i++;
                NormArr[i] = NormSortedTree[i].norm;
            } while (((NormArr[i - 1] - NormArr[i]) / NormArr[i - 1] > 0.1 && i < NormSortedTree.Count - 1) || NormArr[i] > 0.9 * MaxNorm);

            return NormSortedTree[i].norm;
        }

        private double GetNumOfLeafsByNormTH(List<GeoWave> Tree, double WaveletNormTH)
        {

            double counter = 0;
            for (int i = 0; i < Tree.Count; i++)
            {
                if (Tree[i].norm > WaveletNormTH)
                    counter += 1;
            }
            return counter;
        }

        private bool[] getDim2Take(recordConfig rc, int Seed)
        {
            bool[] Dim2Take = new bool[rc.dim];

            var ran = new Random(Seed);

            for (int i = 0; i < rc.NDimsinRF; i++)
            {
                int index = ran.Next(0, rc.dim);
                if (Dim2Take[index] == true)
                    i--;
                else
                    Dim2Take[index] = true;
            }

            return Dim2Take;
        }

        private int getTopLevel(List<GeoWave> decision_GeoWaveArr)
        {
            int topLevel = 0;
            for (int i = 0; i < decision_GeoWaveArr.Count; i++)
                if (decision_GeoWaveArr[i].level > topLevel)
                    topLevel = decision_GeoWaveArr[i].level;
            return topLevel;
        }

        private List<int> Bagging(List<int> trainingArr, double percent, int Seed)//percent in [0,1]
        {
            //List<int> baggedArr = new List<int>();
            int N_rows = Convert.ToInt32(trainingArr.Count * percent);
            //int Seed = (int)DateTime.Now.Ticks;
            var ran = new Random(Seed);
            //            return Enumerable.Range(0, trainingArr.Count).OrderBy(x => ran.Next()).ToList().GetRange(0, N_rows);
            return trainingArr.OrderBy(x => ran.Next()).ToList().GetRange(0, N_rows);
        }

        private List<int> BaggingBreiman(List<int> trainingArr, int Seed)//percent in [0,1]
        {
            bool[] isSet = new bool[trainingArr.Count];
            List<int> baggedArr = new List<int>();
            var ran = new Random(Seed);
            for (int i = 0; i < trainingArr.Count; i++)
            {
                int j = ran.Next(0, trainingArr.Count);
                if (isSet[j] == false)
                    baggedArr.Add(trainingArr[j]);
                isSet[j] = true;
            }
            return baggedArr;
        }

        private int getGWIndexByKfunc(List<GeoWave> tmp_Tree_orderedByNorm,
                                         recordConfig rc,
                                         double[][] trainingData,
                                         double[][] trainingLabel,
                                         ref double best_norm,
                                         List<int> testingArr)
        {
            //double[] best_index_norm = new double[2];//returned value ...
            int NumOfSkips = Convert.ToInt16(1 / rc.NskipsinKfunc);
            int skipSize = Convert.ToInt16(Math.Floor(rc.NskipsinKfunc * tmp_Tree_orderedByNorm.Count));

            if (skipSize * NumOfSkips > tmp_Tree_orderedByNorm.Count)
                MessageBox.Show("skipping made us go out of range - shuold not get here");

            double[] errArr = new double[NumOfSkips - 1];

            ////DO THE HOPPING/SKIPPING
            if (Form1.rumPrallel)
            {
                Parallel.For(1, NumOfSkips, i =>
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    errArr[i - 1] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                });
            }
            else
            {
                for (int i = 1; i < NumOfSkips; i++)
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    errArr[i - 1] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                }
            }

            int best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            int first_index, last_index;
            if (best_index == 0)
            {
                first_index = 0;
                last_index = Math.Min(2 * skipSize, tmp_Tree_orderedByNorm.Count);
            }
            else if (best_index == (NumOfSkips - 2))
            {
                first_index = Math.Max((best_index) * skipSize, 0);
                last_index = tmp_Tree_orderedByNorm.Count;
            }
            else
            {
                first_index = Math.Max((best_index) * skipSize, 0);
                last_index = Math.Min((best_index + 2) * skipSize, tmp_Tree_orderedByNorm.Count);
            }

            errArr = new double[last_index - first_index];

            //SEARCH IN THE BOUNDING 
            if (Form1.rumPrallel)
            {
                Parallel.For(first_index, last_index, i =>
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i].norm;
                    double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i, rc.boostNormsecond, rc.boostTau);
                    errArr[i - first_index] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                });
            }
            else
            {
                for (int i = first_index; i < last_index; i++)
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i].norm;
                    double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i, rc.boostNormsecond, rc.boostTau);
                    errArr[i - first_index] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                }
            }

            best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror
            best_norm = tmp_Tree_orderedByNorm[first_index + best_index].norm;

            return (first_index + best_index);//indicates the number of waveletes to take (calced in order by ID)
        }

        private int getGWIndexByKfuncLessAcurate(List<GeoWave> tmp_Tree_orderedByNorm,
                                         recordConfig rc,
                                         double[][] trainingData,
                                         double[][] trainingLabel,
                                         ref double best_norm,
                                         List<int> testingArr)
        {
            //double[] best_index_norm = new double[2];//returned value ...
            int skipSize = Convert.ToInt16(1 / rc.NskipsinKfunc);
            int NumOfSkips = Convert.ToInt16(Math.Floor(rc.NskipsinKfunc * tmp_Tree_orderedByNorm.Count));

            if (skipSize * NumOfSkips > tmp_Tree_orderedByNorm.Count)
                MessageBox.Show("skipping made us go out of range - shuold not get here");

            double[] errArr = new double[NumOfSkips];

            ////DO THE HOPPING/SKIPPING
            if (Form1.rumPrallel)
            {
                Parallel.For(0, NumOfSkips, i =>
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    if (rc.boostNormsecond == 0)
                        geowave_total_norm += 1;
                    errArr[i] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                });
            }
            else
            {
                for (int i = 0; i < NumOfSkips; i++)
                {
                    double thresholdNorm = tmp_Tree_orderedByNorm[i * skipSize].norm;
                    double Tgt_approx_error = testDecisionTree(testingArr, trainingData, trainingLabel, tmp_Tree_orderedByNorm, thresholdNorm, rc.boostNormTarget);
                    double geowave_total_norm = getgeowaveNorm(tmp_Tree_orderedByNorm, i * skipSize, rc.boostNormsecond, rc.boostTau);
                    if (rc.boostNormsecond == 0)
                        geowave_total_norm += 1;

                    errArr[i] = Tgt_approx_error + (rc.boostlamda_0 * geowave_total_norm);
                }
            }

            int best_index = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            return best_index * skipSize;
        }

        private int getGWIndexByKfuncLessAcuratePooning(List<GeoWave> BoostedTreeArrPooning, recordConfig rc, double[][] training_dt, double[][] boostedLabelsPooning, List<int> testingArr)
        {
            int topLevel = getTopLevel(BoostedTreeArrPooning);
            double[] errorTreeProoning = new double[topLevel];
            double[] NLevels = new double[topLevel];
            double[] errArr = new double[topLevel];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, topLevel, i =>
                {
                    errorTreeProoning[i] = testDecisionTreeWithProoning(testingArr, db.PCAtesting_dt, db.testing_label, BoostedTreeArrPooning, i + 1, 2);
                    NLevels[i] = i;// * rc.hopping_size;
                    errArr[i] = errorTreeProoning[i] + (NLevels[i] * rc.boostProoning_0);
                });
            }
            else
            {
                for (int i = 0; i < topLevel; i++)
                {
                    errorTreeProoning[i] = testDecisionTreeWithProoning(testingArr, db.PCAtesting_dt, db.testing_label, BoostedTreeArrPooning, i + 1, 2);
                    NLevels[i] = i;// *rc.hopping_size;
                    errArr[i] = errorTreeProoning[i] + (NLevels[i] * rc.boostProoning_0);
                }
            }

            int best_level = Enumerable.Range(0, errArr.Length).Aggregate((a, b) => (errArr[a] < errArr[b]) ? a : b); //minerror

            return best_level;
        }

        private void adjustlabels2simplex4(double[][] estimatedLabels)
        {
            double[] dist = new double[4];
            double[][] Data_Lables = new double[4][];
            for (int i = 0; i < 4; i++)
                Data_Lables[i] = new double[3];

            Data_Lables[0][0] = 0; Data_Lables[0][1] = 0; Data_Lables[0][2] = 0; //0 0 0
            Data_Lables[1][0] = 1; Data_Lables[1][1] = 0; Data_Lables[1][2] = 0; //1 0 0 
            Data_Lables[2][0] = 0.5; Data_Lables[2][1] = Math.Sqrt(3) / 2.0; Data_Lables[2][2] = 0;//0.5 sqrt(3)/2 0
            Data_Lables[3][0] = 0.5; Data_Lables[3][1] = Math.Sqrt(3) / 6.0; Data_Lables[3][2] = Math.Sqrt(3) / 6.0; //0.5 sqrt(3)/6 sqrt(3)/6

            for (int i = 0; i < estimatedLabels.Count(); i++)
            {
                for (int k = 0; k < 4; k++)
                    dist[k] = normPoint3d(estimatedLabels[i], Data_Lables[k]);
                int minIndex = Array.IndexOf(dist, dist.Min());
                if (minIndex == 0)
                { estimatedLabels[i][0] = 0; estimatedLabels[i][1] = 0; estimatedLabels[i][2] = 0; }
                else if (minIndex == 1)
                { estimatedLabels[i][0] = 1; estimatedLabels[i][1] = 0; estimatedLabels[i][2] = 0; }
                else if (minIndex == 2)
                { estimatedLabels[i][0] = 0.5; estimatedLabels[i][1] = Math.Sqrt(3) / 2.0; estimatedLabels[i][2] = 0; }
                else
                { estimatedLabels[i][0] = 0.5; estimatedLabels[i][1] = Math.Sqrt(3) / 6.0; estimatedLabels[i][2] = Math.Sqrt(3) / 6.0; }
            }
        }

        private double normPoint3d(double[] p, double[] p_2)
        {
            double norm = 0;
            for (int i = 0; i < p.Count(); i++)
                norm += (p[i] - p_2[i]) * (p[i] - p_2[i]);
            return norm;
        }

        //old version no testarr

        private double testDecisionTree(double[][] Data_table, double[][] Data_Lables, List<GeoWave> Tree_orderedById, double NormThreshold, int NormLPType)
        {
            Tree_orderedById = Tree_orderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[i], Tree_orderedById, NormThreshold);
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[i], Tree_orderedById, NormThreshold);
                }
            }


            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
            }
            else if (NormLPType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeWithProoning(double[][] Data_table, double[][] Data_Lables, List<GeoWave> Tree_orderedById, int topLevel, int NormLPType)
        {
            Tree_orderedById = Tree_orderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(Data_table[i], Tree_orderedById, topLevel);
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(Data_table[i], Tree_orderedById, topLevel);
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
            }
            else if (NormLPType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 1)//L1
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeManyRFNormNbound(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int boundLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, RFdecTreeArr[j], NormThreshold, boundLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, RFdecTreeArr[j], NormThreshold, boundLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            else if (NormLPType == 1)//L1
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }


                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(Data_Lables.Count()));
                }
            }
            else if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[i][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                        {
                            if (Data_Lables[i][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[i][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(Data_Lables.Count()));
                }
            }
            else if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[i][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                        {
                            if (Data_Lables[i][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[i][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(List<int>[] ArrRF_indecesList, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[ArrRF_indecesList[i].Count()][];
                for (int j = 0; j < ArrRF_indecesList[i].Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    //for each tree go over all points
                    for (int j = 0; j < ArrRF_indecesList[i].Count(); j++)
                    {
                        double[] point = new double[rc.dim];
                        for (int t = 0; t < rc.dim; t++)
                            point[t] = double.Parse(Data_table[ArrRF_indecesList[i][j]][t].ToString());
                        double[] tmpLabel = askTreeMeanVal(point, RFdecTreeArr[i], NormThreshold);
                        for (int t = 0; t < Data_Lables[0].Count(); t++)
                        {
                            estimatedLabels[i][j][t] = tmpLabel[t];
                        }
                    }
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    //for each tree go over all points
                    for (int j = 0; j < ArrRF_indecesList[i].Count(); j++)
                    {
                        double[] point = new double[rc.dim];
                        for (int t = 0; t < rc.dim; t++)
                            point[t] = double.Parse(Data_table[ArrRF_indecesList[i][j]][t].ToString());
                        double[] tmpLabel = askTreeMeanVal(point, RFdecTreeArr[i], NormThreshold);
                        for (int t = 0; t < Data_Lables[0].Count(); t++)
                        {
                            estimatedLabels[i][j][t] = tmpLabel[t];
                        }
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < ArrRF_indecesList[k].Count(); i++)//each tree may have diffrent label size
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[ArrRF_indecesList[k][i]][j]) * (estimatedLabels[k][i][j] - Data_Lables[ArrRF_indecesList[k][i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(ArrRF_indecesList[k].Count()));
                }
            }
            else if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < ArrRF_indecesList[k].Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[ArrRF_indecesList[k][i]][j]);
                }
            }
            //else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            //{
            //    double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
            //    for (int k = 0; k < RFdecTreeArr.Count(); k++)
            //    {
            //        double NclassA = 0;
            //        double NclassB = 0;
            //        double NMissclassA = 0;
            //        double NMissclassB = 0;

            //        for (int j = 0; j < Data_Lables[0].Count(); j++)
            //            for (int i = 0; i < Data_Lables.Count(); i++)
            //            {
            //                if (Data_Lables[i][j] == Form1.upper_label)
            //                {
            //                    NclassA += 1;
            //                    if (estimatedLabels[k][i][j] <= threshVal)
            //                        NMissclassA += 1;
            //                }
            //                if (Data_Lables[i][j] == Form1.lower_label)
            //                {
            //                    NclassB += 1;
            //                    if (estimatedLabels[k][i][j] >= threshVal)
            //                        NMissclassB += 1;
            //                }
            //            }
            //        error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            //    }
            //}
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFbyIndex(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int IndexThreshold, int NormLPType)
        {
            List<GeoWave>[] RFdecTreeArrById = new List<GeoWave>[RFdecTreeArr.Count()];
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArrById[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArrById[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], RFdecTreeArr[j][IndexThreshold].norm);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], RFdecTreeArr[j][IndexThreshold].norm);
                    }


                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(Data_Lables.Count()));
                }
            }
            if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[i][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                        {
                            if (Data_Lables[i][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[i][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[Data_Lables.Count()][];
            for (int i = 0; i < Data_Lables.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[i][j]) * (estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(Data_Lables.Count()));
            }
            if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < Data_Lables.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[i][j]);
                    }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    if (Data_Lables[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRF(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(Data_Lables.Count()));
                }
            }
            else if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[i][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                        {
                            if (Data_Lables[i][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[i][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[Data_Lables.Count()][];
                for (int j = 0; j < Data_Lables.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < Data_Lables.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[i][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[i][j]) * (estimatedLabels[k][i][j] - Data_Lables[i][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(Data_Lables.Count()));
                }
            }
            if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[i][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < Data_Lables.Count(); i++)
                        {
                            if (Data_Lables[i][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[i][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeBoosting(double[][] testing_dt, double[][] testing_label, List<GeoWave>[] BoostTreeArr, double NormThreshold, int NormLPType, double[] maxNorms)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, BoostTreeArr.Count(), i =>
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < BoostTreeArr.Count(); i++)
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testing_label.Count()][];
            for (int i = 0; i < testing_label.Count(); i++)
                estimatedLabels[i] = new double[testing_label[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testing_label.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    Parallel.For(0, BoostTreeArr.Count(), j =>
                    {
                        if (NormThreshold < maxNorms[j])
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                        else
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < BoostTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];
                });
            }
            else
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    for (int j = 0; j < BoostTreeArr.Count(); j++)
                    {
                        if (NormThreshold < maxNorms[j])
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                        else
                            tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], NormThreshold);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < BoostTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < testing_label[0].Count(); j++)
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testing_label[i][j]) * (estimatedLabels[i][j] - testing_label[i][j]);
                    }
                error = Math.Sqrt(error);
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testing_label[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], testing_label[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (testing_label[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testing_label[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }
            return error;
        }

        private double testDecisionTreeBoostingLearningRate(double[][] testing_dt, double[][] testing_label, List<GeoWave>[] BoostTreeArr, int NormLPType, double[] maxNorms, int Ntrees, double learningRate)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, Ntrees, i =>
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < Ntrees; i++)
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testing_label.Count()][];
            for (int i = 0; i < testing_label.Count(); i++)
                estimatedLabels[i] = new double[testing_label[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testing_label.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    Parallel.For(0, Ntrees, j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < Ntrees; k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];

                });
            }
            else
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    for (int j = 0; j < Ntrees; j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int k = 0; k < Ntrees; k++)
                            estimatedLabels[i][j] += tmpLabel[k][j];
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < testing_label[0].Count(); j++)
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testing_label[i][j]) * (estimatedLabels[i][j] - testing_label[i][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(estimatedLabels.Count()));
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testing_label[i][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], testing_label[i]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testing_label.Count(); i++)
                {
                    if (testing_label[i][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testing_label[i][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }
            return error;
        }

        private void testDecisionTreeBoostingProoning(double[][] testing_dt, double[][] testing_label, List<GeoWave>[] BoostTreeArrPooning, int[] best_level, int NormLPType, double[] error, double learningRate = 1)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, BoostTreeArrPooning.Count(), i =>
                {
                    BoostTreeArrPooning[i] = BoostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < BoostTreeArrPooning.Count(); i++)
                {
                    BoostTreeArrPooning[i] = BoostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[testing_label.Count()][][];
            for (int i = 0; i < testing_label.Count(); i++)
            {
                estimatedLabels[i] = new double[rc.boostNum][];
                for (int j = 0; j < rc.boostNum; j++)
                    estimatedLabels[i][j] = new double[testing_label[0].Count()];
            }


            if (Form1.rumPrallel)
            {
                Parallel.For(0, testing_label.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArrPooning.Count()][];
                    Parallel.For(0, BoostTreeArrPooning.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, BoostTreeArrPooning[j], best_level[j]);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < BoostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + (learningRate * tmpLabel[k][j]);
                            tmp += (learningRate * tmpLabel[k][j]);
                        }
                    }
                });
            }
            else
            {
                for (int i = 0; i < testing_label.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[i][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArrPooning.Count()][];
                    for (int j = 0; j < BoostTreeArrPooning.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, BoostTreeArrPooning[j], best_level[j]);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < BoostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + (learningRate * tmpLabel[k][j]);
                            tmp += (learningRate * tmpLabel[k][j]);
                        }
                    }
                }
            }

            if (NormLPType == 2)
            {
                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int i = 0; i < testing_label.Count(); i++)
                            error[k] += (estimatedLabels[i][k][j] - testing_label[i][j]) * (estimatedLabels[i][k][j] - testing_label[i][j]);
                    error[k] = Math.Sqrt(error[k]);
                }

            }
            else if (NormLPType == 0 && estimatedLabels[0][0].Count() == 1)//+-1 labels
            {
                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        if ((estimatedLabels[i][k][0] * testing_label[i][0]) <= 0)
                            error[k] += 1;
                    }
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double[] NclassA = new double[rc.boostNum];
                double[] NclassB = new double[rc.boostNum];
                double[] NMissclassA = new double[rc.boostNum];
                double[] NMissclassB = new double[rc.boostNum];

                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        if (testing_label[i][0] == 1)
                        {
                            NclassA[k] += 1;
                            if (estimatedLabels[i][k][0] <= 0)
                                NMissclassA[k] += 1;
                        }
                        if (testing_label[i][0] == -1)
                        {
                            NclassB[k] += 1;
                            if (estimatedLabels[i][k][0] >= 0)
                                NMissclassB[k] += 1;
                        }
                    }
                    error[k] = 0.5 * ((NMissclassA[k] / NclassA[k]) + (NMissclassB[k] / NclassB[k]));
                }
            }
            //else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            //{
            //    //adjust labels to simplex
            //    adjustlabels2simplex4(estimatedLabels);

            //    for (int i = 0; i < Data_Lables.Count(); i++)
            //    {
            //        if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[i]))
            //            error += 1;
            //    }
            //}

        }


        //end old version no testarr

        private double testDecisionTree(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave> Tree_orderedById, double NormThreshold, int NormLPType)
        {
            Tree_orderedById = Tree_orderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[testingArr[i]], Tree_orderedById, NormThreshold);
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanVal(Data_table[testingArr[i]], Tree_orderedById, NormThreshold);
                }
            }


            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
            }
            else if (NormLPType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (Data_Lables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeWithProoning(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave> Tree_orderedById, int topLevel, int NormLPType)
        {
            Tree_orderedById = Tree_orderedById.OrderBy(o => o.ID).ToList();

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(Data_table[testingArr[i]], Tree_orderedById, topLevel);
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    estimatedLabels[i] = askTreeMeanValAtLevel(Data_table[testingArr[i]], Tree_orderedById, topLevel);
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
            }
            else if (NormLPType == -1)//max
            {
                List<double> errList = new List<double>();
                double tmp = 0;
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    tmp = 0;
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        tmp += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                    errList.Add(tmp);
                }
                error = errList.Max();
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (Data_Lables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeRF(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (NormLPType == 1)//L1
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (Data_Lables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeManyRFNormNbound(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int boundLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, RFdecTreeArr[j], NormThreshold, boundLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValBoundLevel(point, RFdecTreeArr[j], NormThreshold, boundLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            else if (NormLPType == 1)//L1
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (Data_Lables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRF(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }


                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            else if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (Data_Lables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, double NormThreshold, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArr[j], NormThreshold);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            else if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (Data_Lables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFbyIndex(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int IndexThreshold, int NormLPType)
        {
            List<GeoWave>[] RFdecTreeArrById = new List<GeoWave>[RFdecTreeArr.Count()];
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArrById[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArrById[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, Data_Lables.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], RFdecTreeArr[j][IndexThreshold].norm);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, RFdecTreeArrById[j], RFdecTreeArr[j][IndexThreshold].norm);
                    }


                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (Data_Lables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double testDecisionTreeRF(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[Data_Lables[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[i][j] += tmpLabel[k][j] / Convert.ToDouble(RFdecTreeArr.Count());
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(testingArr.Count()));
            }
            if (NormLPType == 1)
            {
                for (int j = 0; j < Data_Lables[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += Math.Abs(estimatedLabels[i][j] - Data_Lables[testingArr[i]][j]);
                    }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * Data_Lables[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() > 1)//3d simplex
            {
                //adjust labels to simplex
                adjustlabels2simplex4(estimatedLabels);

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (0.00001 < normPoint3d(estimatedLabels[i], Data_Lables[testingArr[i]]))
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (Data_Lables[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (Data_Lables[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRF(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        estimatedLabels[0][i][j] = tmpLabel[0][j];
                        for (int k = 1; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = (Convert.ToDouble(k) / (Convert.ToDouble(k) + 1)) * estimatedLabels[k - 1][i][j] + (1 / (Convert.ToDouble(k) + 1)) * tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            else if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (Data_Lables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private double[] testDecisionTreeManyRFNoVoting(List<int> testingArr, double[][] Data_table, double[][] Data_Lables, List<GeoWave>[] RFdecTreeArr, int topLevel, int NormLPType)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    RFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[RFdecTreeArr.Count()][][];//num of trees, label index, label values (or value in most cases)
            for (int i = 0; i < RFdecTreeArr.Count(); i++)
            {
                estimatedLabels[i] = new double[testingArr.Count()][];
                for (int j = 0; j < testingArr.Count(); j++)
                    estimatedLabels[i][j] = new double[Data_Lables[0].Count()];
            }

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];
                    Parallel.For(0, RFdecTreeArr.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    });

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//Data_table[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//Data_table[0].Count()
                        point[j] = double.Parse(Data_table[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[RFdecTreeArr.Count()][];

                    for (int j = 0; j < RFdecTreeArr.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, RFdecTreeArr[j], topLevel);
                    }

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                    {
                        for (int k = 0; k < RFdecTreeArr.Count(); k++)
                            estimatedLabels[k][i][j] = tmpLabel[k][j];
                    }
                }
            }

            double[] error = new double[RFdecTreeArr.Count()];
            if (NormLPType == 2)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]) * (estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }
            }
            if (NormLPType == 1)
            {
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                            error[k] += Math.Abs(estimatedLabels[k][i][j] - Data_Lables[testingArr[i]][j]);
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double threshVal = 0.5 * (Form1.upper_label + Form1.lower_label);
                for (int k = 0; k < RFdecTreeArr.Count(); k++)
                {
                    double NclassA = 0;
                    double NclassB = 0;
                    double NMissclassA = 0;
                    double NMissclassB = 0;

                    for (int j = 0; j < Data_Lables[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)
                        {
                            if (Data_Lables[testingArr[i]][j] == Form1.upper_label)
                            {
                                NclassA += 1;
                                if (estimatedLabels[k][i][j] <= threshVal)
                                    NMissclassA += 1;
                            }
                            if (Data_Lables[testingArr[i]][j] == Form1.lower_label)
                            {
                                NclassB += 1;
                                if (estimatedLabels[k][i][j] >= threshVal)
                                    NMissclassB += 1;
                            }
                        }
                    error[k] = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
                }
            }
            return error;

            //printErrorsToFile(Form1.MainFolderName + Form1.dataStruct[5] + "\\misslabeling_results_Dim" + test_table_low_dim.Columns.Count.ToString() + ".txt", l2_error, l1_error, numOfMissLables, test_Lables.Rows.Count);
        }

        private void testDecisionTreeBoostingProoning(List<int> testingArr, double[][] testing_dt, double[][] testing_label,
            List<GeoWave>[] BoostTreeArrPooning, int[] best_level, int NormLPType, double[] error, double learningRate = 1)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, BoostTreeArrPooning.Count(), i =>
                {
                    BoostTreeArrPooning[i] = BoostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < BoostTreeArrPooning.Count(); i++)
                {
                    BoostTreeArrPooning[i] = BoostTreeArrPooning[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][][] estimatedLabels = new double[testingArr.Count()][][];//testing_label
            for (int i = 0; i < testingArr.Count(); i++)//testing_label
            {
                estimatedLabels[i] = new double[rc.boostNum][];
                for (int j = 0; j < rc.boostNum; j++)
                    estimatedLabels[i][j] = new double[testing_label[0].Count()];
            }


            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>     //testing_label
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[testingArr[i]][j].ToString());//[i][j]
                    double[][] tmpLabel = new double[BoostTreeArrPooning.Count()][];
                    Parallel.For(0, BoostTreeArrPooning.Count(), j =>
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, BoostTreeArrPooning[j], best_level[j]);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < BoostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + (learningRate * tmpLabel[k][j]);
                            tmp += (learningRate * tmpLabel[k][j]);
                        }
                    }
                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)//testing_label
                {
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[testingArr[i]][j].ToString());//[i][j]
                    double[][] tmpLabel = new double[BoostTreeArrPooning.Count()][];
                    for (int j = 0; j < BoostTreeArrPooning.Count(); j++)
                    {
                        tmpLabel[j] = askTreeMeanValAtLevel(point, BoostTreeArrPooning[j], best_level[j]);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        double tmp = 0;
                        for (int k = 0; k < BoostTreeArrPooning.Count(); k++)
                        {
                            estimatedLabels[i][k][j] = tmp + (learningRate * tmpLabel[k][j]);
                            tmp += (learningRate * tmpLabel[k][j]);
                        }
                    }
                }
            }

            if (NormLPType == 2)
            {
                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int j = 0; j < testing_label[0].Count(); j++)
                        for (int i = 0; i < testingArr.Count(); i++)//testing_label
                            error[k] += (estimatedLabels[i][k][j] - testing_label[testingArr[i]][j]) * (estimatedLabels[i][k][j] - testing_label[testingArr[i]][j]);//[i][j]
                    error[k] = Math.Sqrt(error[k] / Convert.ToDouble(testingArr.Count()));
                }

            }
            else if (NormLPType == 0 && estimatedLabels[0][0].Count() == 1)//+-1 labels
            {
                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int i = 0; i < testingArr.Count(); i++)//testing_label
                    {
                        if ((estimatedLabels[i][k][0] * testing_label[testingArr[i]][0]) <= 0)//[i][0]
                            error[k] += 1;
                    }
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0][0].Count() == 1)//+-1 labels + BER
            {
                double[] NclassA = new double[rc.boostNum];
                double[] NclassB = new double[rc.boostNum];
                double[] NMissclassA = new double[rc.boostNum];
                double[] NMissclassB = new double[rc.boostNum];

                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int i = 0; i < testing_label.Count(); i++)
                    {
                        if (testing_label[testingArr[i]][0] == 1)//[i][0]
                        {
                            NclassA[k] += 1;
                            if (estimatedLabels[i][k][0] <= 0)
                                NMissclassA[k] += 1;
                        }
                        if (testing_label[testingArr[i]][0] == -1)//[i][0]
                        {
                            NclassB[k] += 1;
                            if (estimatedLabels[i][k][0] >= 0)
                                NMissclassB[k] += 1;
                        }
                    }
                    error[k] = 0.5 * ((NMissclassA[k] / NclassA[k]) + (NMissclassB[k] / NclassB[k]));
                }
            }
            else if (NormLPType == -10) // clasification
            {
                double[] sort_testing_label = new double[testing_label.Count()];
                double[] uniqe_testing_label = new double[testing_label.Count()];
                double min_diff = 999;
                int counter = 0;
                int closest_ind = -1;

                for (int ii = 0; ii < testing_label.Count(); ii++)
                {
                    sort_testing_label[ii] = testing_label[ii][0];
                }

                Array.Sort(sort_testing_label);
                uniqe_testing_label[0] = sort_testing_label[0];
                for (int ii = 1; ii < testing_label.Count(); ii++)
                {
                    if (sort_testing_label[ii] != sort_testing_label[ii - 1])
                    {
                        counter++;
                        uniqe_testing_label[counter] = sort_testing_label[ii];
                    }
                }

                double[] cut_uniqe_testing_label3 = new double[counter + 1];

                for (int ii = 0; ii <= counter; ii++)
                {
                    cut_uniqe_testing_label3[ii] = uniqe_testing_label[ii];
                }

                for (int k = 0; k < rc.boostNum; k++)
                {
                    for (int i = 0; i < testingArr.Count(); i++) //testing_label
                    {
                        min_diff = 9999;
                        for (int j = 0; j < cut_uniqe_testing_label3.Count(); j++)
                        {
                            if (Math.Abs(estimatedLabels[i][k][0] - cut_uniqe_testing_label3[j]) < min_diff)
                            {
                                min_diff = estimatedLabels[i][k][0] - cut_uniqe_testing_label3[j];
                                closest_ind = j;
                            }
                        }

                        if (cut_uniqe_testing_label3[closest_ind] == testing_label[testingArr[i]][0])//[i][0]
                            error[k] += 1;
                    }
                    error[k] = error[k] / testingArr.Count;

                }
            }
        }

        private double testDecisionTreeBoostingLearningRate(List<int> testingArr, double[][] testing_dt, double[][] testing_label,
            List<GeoWave>[] BoostTreeArr, int NormLPType, double[] maxNorms, int Ntrees, double learningRate)
        {
            if (Form1.rumPrallel)
            {
                Parallel.For(0, Ntrees, i =>
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                });
            }
            else
            {
                for (int i = 0; i < Ntrees; i++)
                {
                    BoostTreeArr[i] = BoostTreeArr[i].OrderBy(o => o.ID).ToList();
                }
            }

            //GO OVER TESTING DATA AND GET ESTIMATIONS FOR EACH DATA LINE
            double[][] estimatedLabels = new double[testingArr.Count()][];
            for (int i = 0; i < testingArr.Count(); i++)
                estimatedLabels[i] = new double[testing_label[0].Count()];

            if (Form1.rumPrallel)
            {
                Parallel.For(0, testingArr.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    Parallel.For(0, Ntrees, j =>
                    {
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                    });

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        //estimatedLabels[i][j] += tmpLabel[0][j]; //FIRST TREE
                        for (int k = 0; k < Ntrees; k++)//k=1
                            estimatedLabels[i][j] += (learningRate * tmpLabel[k][j]);//REST OF TREES
                    }

                });
            }
            else
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//testing_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//testing_dt[0].Count()
                        point[j] = double.Parse(testing_dt[testingArr[i]][j].ToString());
                    double[][] tmpLabel = new double[BoostTreeArr.Count()][];
                    for (int j = 0; j < Ntrees; j++)
                    {
                        tmpLabel[j] = askTreeMeanVal(point, BoostTreeArr[j], maxNorms[j]);
                    }

                    for (int j = 0; j < testing_label[0].Count(); j++)
                    {
                        //estimatedLabels[i][j] += tmpLabel[0][j]; //FIRST TREE
                        for (int k = 0; k < Ntrees; k++)//k=1
                            estimatedLabels[i][j] += (learningRate * tmpLabel[k][j]);//REST OF TREES
                    }
                }
            }

            double error = 0;
            if (NormLPType == 2)
            {
                for (int j = 0; j < testing_label[0].Count(); j++)
                    for (int i = 0; i < testingArr.Count(); i++)
                    {
                        error += (estimatedLabels[i][j] - testing_label[testingArr[i]][j]) * (estimatedLabels[i][j] - testing_label[testingArr[i]][j]);
                    }
                error = Math.Sqrt(error / Convert.ToDouble(estimatedLabels.Count()));
            }
            else if (NormLPType == 0 && estimatedLabels[0].Count() == 1)//+-1 labels
            {
                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if ((estimatedLabels[i][0] * testing_label[testingArr[i]][0]) <= 0)
                        error += 1;
                }
            }
            else if (NormLPType == -2 && estimatedLabels[0].Count() == 1)//+-1 labels + BER
            {
                double NclassA = 0;
                double NclassB = 0;
                double NMissclassA = 0;
                double NMissclassB = 0;

                for (int i = 0; i < testingArr.Count(); i++)
                {
                    if (testing_label[testingArr[i]][0] == 1)
                    {
                        NclassA += 1;
                        if (estimatedLabels[i][0] <= 0)
                            NMissclassA += 1;
                    }
                    if (testing_label[testingArr[i]][0] == -1)
                    {
                        NclassB += 1;
                        if (estimatedLabels[i][0] >= 0)
                            NMissclassB += 1;
                    }
                }
                error = 0.5 * ((NMissclassA / NclassA) + (NMissclassB / NclassB));
            }

            else if (NormLPType == -10) // clasification
            {
                double[] sort_testing_label = new double[testing_label.Count()];
                double[] uniqe_testing_label = new double[testing_label.Count()];
                double min_diff = 999;
                int counter = 0;
                int closet_ind = 0;

                for (int ii = 0; ii < testing_label.Count(); ii++)
                {
                    sort_testing_label[ii] = testing_label[ii][0];
                }

                Array.Sort(sort_testing_label);
                uniqe_testing_label[0] = sort_testing_label[0];
                for (int ii = 1; ii < testing_label.Count(); ii++)
                {
                    if (sort_testing_label[ii] != sort_testing_label[ii - 1])
                    {
                        counter++;
                        uniqe_testing_label[counter] = sort_testing_label[ii];
                    }
                }

                double[] cut_uniqe_testing_label3 = new double[counter + 1];

                for (int ii = 0; ii <= counter; ii++)
                {
                    cut_uniqe_testing_label3[ii] = uniqe_testing_label[ii];
                }
                for (int i = 0; i < testingArr.Count(); i++) //testing_label
                {
                    min_diff = 9999;
                    for (int j = 0; j < cut_uniqe_testing_label3.Count(); j++)
                    {
                        if (Math.Abs(estimatedLabels[i][0] - cut_uniqe_testing_label3[j]) < min_diff)
                        {
                            min_diff = estimatedLabels[i][0] - cut_uniqe_testing_label3[j];
                            closet_ind = j;
                        }
                    }

                    if (cut_uniqe_testing_label3[closet_ind] == testing_label[testingArr[i]][0])//[i][0]
                        error += 1;
                }
                error = error / testingArr.Count;
            }
            return error;
        }

        private List<GeoWave>[] getsparseRF(List<GeoWave>[] RFdecTreeArr, int Nwavelets)
        {
            List<GeoWave>[] sparseRF = new List<GeoWave>[RFdecTreeArr.Count()];
            bool[][] wasElementSet = new bool[RFdecTreeArr.Count()][];
            List<GeoWave>[] IDRFdecTreeArr = new List<GeoWave>[RFdecTreeArr.Count()];

            if (Form1.rumPrallel)
            {
                //FOR EACH i TREE 
                Parallel.For(0, RFdecTreeArr.Count(), i =>
                {
                    IDRFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                    wasElementSet[i] = new bool[RFdecTreeArr[i].Count];
                    sparseRF[i] = new List<GeoWave>();

                    //SET WAVELETS
                    int Loops = (RFdecTreeArr[i].Count > Nwavelets) ? Nwavelets : RFdecTreeArr[i].Count;//set loops = min(Nwavelets, RFdecTreeArr[j].Count); 
                    for (int j = 0; j < Loops; j++)//each wavelets (up till Loops)
                    {
                        //COPY WAVELET - IF WAS NOT COPIED BEFORE
                        if (wasElementSet[i][RFdecTreeArr[i][j].ID] == false)
                        {
                            sparseRF[i].Add(RFdecTreeArr[i][j]);
                            wasElementSet[i][RFdecTreeArr[i][j].ID] = true;
                        }

                        int parentID = RFdecTreeArr[i][j].parentID;
                        while (parentID != -1)
                        {
                            //COPY PARENT WAVELET - IF WAS NOT COPIED BEFORE
                            if (wasElementSet[i][parentID] == false)
                            {
                                sparseRF[i].Add(IDRFdecTreeArr[i][parentID]);
                                wasElementSet[i][parentID] = true;
                            }
                            parentID = IDRFdecTreeArr[i][parentID].parentID;
                        }
                    }

                    //SORT
                    sparseRF[i] = sparseRF[i].OrderByDescending(o => o.norm).ToList();

                    Dictionary<int, int> IDmap = new Dictionary<int, int>();//old ID, new ID

                    //NULLIFY CHILDREN OF REMOVED WAVELETS
                    for (int j = 0; j < sparseRF[i].Count; j++)
                    {
                        IDmap.Add(sparseRF[i][j].ID, j);
                        sparseRF[i][j].ID = j;

                        if (sparseRF[i][j].child0 != -1 && wasElementSet[i][sparseRF[i][j].child0] == false)
                            sparseRF[i][j].child0 = -1;
                        if (sparseRF[i][j].child1 != -1 && wasElementSet[i][sparseRF[i][j].child1] == false)
                            sparseRF[i][j].child1 = -1;
                    }

                    //SET NEW ID
                    for (int j = 0; j < sparseRF[i].Count; j++)
                    {
                        int newID;
                        if (IDmap.TryGetValue(sparseRF[i][j].child0, out newID))
                            sparseRF[i][j].child0 = newID;
                        if (IDmap.TryGetValue(sparseRF[i][j].child1, out newID))
                            sparseRF[i][j].child1 = newID;
                    }

                });
            }
            else
            {
                //SORT RF TREES
                for (int i = 0; i < RFdecTreeArr.Count(); i++)
                {
                    IDRFdecTreeArr[i] = RFdecTreeArr[i].OrderBy(o => o.ID).ToList();
                    wasElementSet[i] = new bool[RFdecTreeArr[i].Count];
                    sparseRF[i] = new List<GeoWave>();

                    //SET WAVELETS
                    int Loops = (RFdecTreeArr[i].Count > Nwavelets) ? Nwavelets : RFdecTreeArr[i].Count;//set loops = min(Nwavelets, RFdecTreeArr[j].Count); 
                    for (int j = 0; j < Loops; j++)//each wavelets (up till Loops)
                    {
                        //COPY WAVELET - IF WAS NOT COPIED BEFORE
                        if (wasElementSet[i][RFdecTreeArr[i][j].ID] == false)
                        {
                            sparseRF[i].Add(RFdecTreeArr[i][j]);
                            wasElementSet[i][RFdecTreeArr[i][j].ID] = true;
                        }

                        int parentID = RFdecTreeArr[i][j].parentID;
                        while (parentID != -1)
                        {
                            //COPY PARENT WAVELET - IF WAS NOT COPIED BEFORE
                            if (wasElementSet[i][parentID] == false)
                            {
                                sparseRF[i].Add(IDRFdecTreeArr[i][parentID]);
                                wasElementSet[i][parentID] = true;
                            }
                            parentID = IDRFdecTreeArr[i][parentID].parentID;
                        }
                    }

                    //SORT
                    sparseRF[i] = sparseRF[i].OrderByDescending(o => o.norm).ToList();

                    //NULLIFY CHILDREN OF REMOVED WAVELETS
                    for (int j = 0; j < sparseRF[i].Count; j++)
                    {
                        if (sparseRF[i][j].child0 != -1 && wasElementSet[i][sparseRF[i][j].child0] == false)
                            sparseRF[i][j].child0 = -1;
                        if (sparseRF[i][j].child1 != -1 && wasElementSet[i][sparseRF[i][j].child1] == false)
                            sparseRF[i][j].child1 = -1;
                    }
                }
            }
            return sparseRF;
        }

        private double getgeowaveNorm(List<GeoWave> tmp_Tree_orderedByNorm, int Nwavelets, int NormSecond, int orderTau)
        {
            double norm = 0;
            if (NormSecond == 0)
                return 1.0 * Nwavelets;// I dont add +1 because if I root estimation I want to give norm 0
            else if (orderTau == 1)
            {
                for (int i = 0; i <= Nwavelets; i++)
                    norm += tmp_Tree_orderedByNorm[i].norm;
                return norm;
            }
            else if (orderTau == 2)
            {
                for (int i = 0; i <= Nwavelets; i++)
                    norm += (tmp_Tree_orderedByNorm[i].norm) * (tmp_Tree_orderedByNorm[i].norm);
                return Math.Sqrt(norm);
            }
            else
            {
                for (int i = 0; i <= Nwavelets; i++)
                    norm += Math.Pow(tmp_Tree_orderedByNorm[i].norm, orderTau);
                return Math.Pow(norm, 1 / Convert.ToDouble(orderTau));
            }
        }

        private double[] askTreeMeanVal(double[] point, List<GeoWave> Tree_orderedById, double NormThreshold)
        {

            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim))
            {
                DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];
            // the next two line is writed by ohadmo: for my opinion - correction for bug in oren's code. the function cant return 0. if the correct answer is 0, the function returns the root mean value.
            for (int i = 0; i < zeroMean.Count(); i++)
                zeroMean[i] = 0;

            double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;
            bool endOfLoop = false;

            while (!endOfLoop)
            {
                if (Tree_orderedById[parent_index].child0 != -1 && DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else if (Tree_orderedById[parent_index].child1 != -1 && DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child1].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
                else
                    endOfLoop = true;
            }
            return MeanValue;
        }

        private double[] askTreeMeanVal_deepestVal(double[] point, List<GeoWave> Tree_orderedById, double NormThreshold)
        {

            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim))
            {
                DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];
            // the next two line is writed by ohadmo: for my opinion - correction for bug in oren's code. the function cant return 0. if the correct answer is 0, the function returns the root mean value.
            for (int i = 0; i < zeroMean.Count(); i++)
                zeroMean[i] = 0;

            double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;
            bool endOfLoop = false;

            while (!endOfLoop)
            {
                if (Tree_orderedById[parent_index].child0 != -1 && DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        MeanValue[0] = Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue[0];
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else if (Tree_orderedById[parent_index].child1 != -1 && DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child1].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        MeanValue[0] = Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue[0];
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
                else
                    endOfLoop = true;
            }
            return MeanValue;
        }

        private double[] askTreeMeanValBoundLevel(double[] point, List<GeoWave> Tree_orderedById, double NormThreshold, int BoundLevel)
        {
            //if (point.Count() != Tree_orderedById[0].boubdingBox[0].Count())
            //{
            //    MessageBox.Show("the dim of the point is not compatible with the dim of the tree");
            //    return null;
            //}

            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim))
            {
                DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];
            double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];
            // the next two line is writed by ohadmo: for my opinion - correction for bug in oren's code. the function cant return 0. if the correct answer is 0, the function returns the root mean value.
            for (int i = 0; i < zeroMean.Count(); i++)
                zeroMean[i] = 555.666;


            //SET THE ROOT MEAN VAL
            Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;

            while (Tree_orderedById[parent_index].child0 != -1 && Tree_orderedById[parent_index].level <= BoundLevel)
            {
                if (DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child0].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        NormThreshold <= Tree_orderedById[Tree_orderedById[parent_index].child1].norm) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        //Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                        MeanValue[0] += (Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue[0] - Tree_orderedById[parent_index].MeanValue[0]);
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
            }
            return MeanValue;
        }

        private double[] askTreeMeanValAtLevel(double[] point, List<GeoWave> Tree_orderedById, int topLevel)
        {
            int counter = 0;
            if (!DB.IsPntInsideBox(Tree_orderedById[0].boubdingBox, point, rc.dim)) // אם הנקודה לא נמצאת בעץ בכלל, נטיל אותה על העץ - כניראה למניעת באגים אח"כ
            {
                DB.ProjectPntInsideBox(Tree_orderedById[0].boubdingBox, ref point);
                counter++;
            }

            double[] zeroMean = new double[Tree_orderedById[0].MeanValue.Count()];

            // the next two line is writed by ohadmo: for my opinion - correction for bug in oren's code. the function cant return 0. if the correct answer is 0, the function returns the root mean value.
            for (int i = 0; i < zeroMean.Count(); i++)
                zeroMean[i] = 555.666;


            double[] MeanValue = new double[Tree_orderedById[0].MeanValue.Count()];

            //SET THE ROOT MEAN VAL
            Tree_orderedById[0].MeanValue.CopyTo(MeanValue, 0);

            ////get to leaf 

            int parent_index = 0;


            // I MUST "REPAIR" THIS WHILE - NEED TO STOP IT WHEN TOP LEVEL REACHED.
            while (Tree_orderedById[parent_index].level < topLevel && Tree_orderedById[parent_index].child0 != -1) // if 
            {
                if (DB.IsPntInsideBox(Tree_orderedById[Tree_orderedById[parent_index].child0].boubdingBox, point, rc.dim))
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.SequenceEqual(zeroMean) &&
                        topLevel >= Tree_orderedById[Tree_orderedById[parent_index].child0].level) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        Tree_orderedById[Tree_orderedById[parent_index].child0].MeanValue.CopyTo(MeanValue, 0);
                    }

                    parent_index = Tree_orderedById[parent_index].child0;
                }
                else
                {
                    if (!Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.SequenceEqual(zeroMean) &&
                        topLevel >= Tree_orderedById[Tree_orderedById[parent_index].child1].level) //take the mean value if its not 0 and the wavelete should be taken ( norm size) - or if its the root wavelete
                    {
                        Tree_orderedById[Tree_orderedById[parent_index].child1].MeanValue.CopyTo(MeanValue, 0);
                    }

                    parent_index = Tree_orderedById[parent_index].child1;
                }
            }
            return MeanValue;
        }

        public double[][] GetResidualLabelsInBoosting(List<GeoWave> Tree, double[][] training_dt, double[][] boostedLabels, double threshNorm)
        {
            List<GeoWave> Tree_orderedById = Tree.OrderBy(o => o.ID).ToList();

            if (Form1.rumPrallel)
            {
                Parallel.For(0, boostedLabels.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        boostedLabels[i][j] = boostedLabels[i][j] - tmpLabel[j];
                });
            }
            else
            {
                for (int i = 0; i < boostedLabels.Count(); i++)
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        boostedLabels[i][j] = boostedLabels[i][j] - tmpLabel[j];
                }
            }

            return boostedLabels;
        }

        public double[][] GetResidualLabelsInBoostingLearningRate(List<GeoWave> Tree, double[][] training_dt, double[][] boostedLabels, double threshNorm, double LearningRate)
        {
            double[][] res_boostedLabels = new double[boostedLabels.Count()][];
            for (int i = 0; i < boostedLabels.Count(); i++)
                res_boostedLabels[i] = new double[boostedLabels[i].Count()];


            List<GeoWave> Tree_orderedById = Tree.OrderBy(o => o.ID).ToList();

            if (Form1.rumPrallel)
            {
                Parallel.For(0, boostedLabels.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        res_boostedLabels[i][j] = boostedLabels[i][j] - LearningRate * tmpLabel[j];
                });
            }
            else
            {
                for (int i = 0; i < boostedLabels.Count(); i++)
                {
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanVal(point, Tree_orderedById, threshNorm);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        res_boostedLabels[i][j] = boostedLabels[i][j] - LearningRate * tmpLabel[j];
                }
            }

            return res_boostedLabels;
        }

        private double[][] GetResidualLabelsInBoostingProoning(List<GeoWave> Tree, double[][] training_dt, double[][] boostedLabelsPooning, int best_level, double learningRate = 1)
        {
            double[][] res_boostedLabels = new double[boostedLabelsPooning.Count()][];
            for (int i = 0; i < boostedLabelsPooning.Count(); i++)
                res_boostedLabels[i] = new double[boostedLabelsPooning[i].Count()];

            List<GeoWave> Tree_orderedById = Tree.OrderBy(o => o.ID).ToList();

            if (Form1.rumPrallel)
            {
                Parallel.For(0, boostedLabelsPooning.Count(), i =>
                {
                    //test_table_low_dim.Rows[i].ToArray().CopyTo(point,0);
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    //Data_table.CopyTo(point, i);
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanValAtLevel(point, Tree_orderedById, best_level);
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        res_boostedLabels[i][j] = boostedLabelsPooning[i][j] - learningRate * tmpLabel[j];
                });
            }
            else
            {
                for (int i = 0; i < boostedLabelsPooning.Count(); i++)
                {
                    double[] point = new double[rc.dim];//training_dt[0].Count()
                    for (int j = 0; j < rc.dim; j++)//training_dt[0].Count()
                        point[j] = double.Parse(training_dt[i][j].ToString());

                    double[] tmpLabel = askTreeMeanValAtLevel(point, Tree_orderedById, best_level); // מחזיר את הערך הממוצע של העלה הנכון עד לרמה המבוקשת בעץ שחושבs
                    for (int j = 0; j < tmpLabel.Count(); j++)
                        res_boostedLabels[i][j] = boostedLabelsPooning[i][j] - learningRate * tmpLabel[j];
                }
            }

            return res_boostedLabels;
        }

        public static void printErrorsOfTree(double[] errArr, string filename)
        {
            StreamWriter writer;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else
                writer = new StreamWriter(filename, false);

            for (int i = 0; i < errArr.Count(); i++)
                writer.WriteLine(errArr[i]);
            writer.Close();
        }

        public static void printTotalResidual(double[] TotalRsidual, string filename)
        {
            StreamWriter writer;

            writer = new StreamWriter(filename, false);

            for (int i = 0; i < TotalRsidual.Count(); i++)
                writer.WriteLine(TotalRsidual[i]);
            writer.Close();
        }

        public static void printErrorsOfTree(double[] errArr, double[] NwavesArr, string filename)
        {
            StreamWriter writer;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else
                writer = new StreamWriter(filename, false);
            for (int i = 0; i < errArr.Count(); i++)
                writer.WriteLine(NwavesArr[i] + " " + errArr[i]);
            writer.Close();
        }

        public static void printErrorsOfTree(double err, int Nwaves, string filename)
        {
            StreamWriter writer;
            if (Form1.UseS3)
            {
                string dir_name = Path.GetDirectoryName(filename);
                string file_name = Path.GetFileName(filename);

                S3DirectoryInfo s3dir = new S3DirectoryInfo(Form1.S3client, Form1.bucketName, dir_name);
                S3FileInfo artFile = s3dir.GetFile(file_name);
                writer = new StreamWriter(artFile.OpenWrite());
            }
            else
                writer = new StreamWriter(filename, false);

            writer.WriteLine(Nwaves + " " + err);
            writer.Close();
        }

    }

}