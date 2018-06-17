package opencv;

import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.*;
import java.lang.Math;
import java.lang.Object;
import java.util.*;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import java.util.LinkedList;
import java.io.File;
import java.math.BigDecimal;
import java.lang.Number;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.File;
import java.io.PrintStream;
import java.util.Random;
import java.util.logging.Logger;

public class TestMultiLayerPerceptron {
	 /**
     * 多層パーセプトロンの実装
     * @author karura
     * @param args
     */
    public static void main(String[] args){
        new TestMultiLayerPerceptron();
    }
     
    /**
     * 処理関数
     */
    public TestMultiLayerPerceptron(){
    	int num = 500;
    	int n = 19;
    	String[][] data = new String[num][n];
    	int x1,x2,k;
    	int l = 3;
    	final double[][] x        = new double [num][n-1];
        final double[]   answer   = new double [num];
        double [][] kari = new double [num][n];
    	// OR計算の教師データ
        // 入力データ配列 x =(入力1,入力2)の配列と,正解データ配列 answer
    	/*final double[][] x        = { { 1.0f , 1.0f ,1.0f} , 
                { 1.0f , 0.0f ,0.0f} , 
                { 0.0f , 1.0f ,0.0f} , 
                { 0.0f , 0.0f ,0.0f}  };
final double[]   answer   = {  0.0f ,
                 1.0f ,
                 1.0f ,
                 0.0f };*/
    	
        
    	try{
			File f = new File("butterfly3seiki.csv");
			//File f = new File("test.csv");
			BufferedReader br = new BufferedReader(new FileReader(f));

			String line = br.readLine();

			for (int row = 0; line != null; row++) {
				data[row] = line.split(",", 0);
				line = br.readLine();
			}
			br.close();
		}catch (IOException a){
			//System.out.println(a);
		}
    	for(x1 = 0; x1 < num; x1++){
			for(x2 = 0; x2 < n; x2++){
				kari[x1][x2] = Double.parseDouble(data[x1][x2]);
			}
		}
    	//正解のナンバー振り
    	/*for(x1 = 0; x1 < num; x1++){
    		if(kari[x1][18] == 1){
    			answer[x1] = 0.0;
    		}else if(kari[x1][18] == 2){
    			answer[x1] = 0.5;
    		}else if(kari[x1][18] == 3){
    			answer[x1] = 1.0;
    		}
    	}*/
    	for(x1 = 0; x1 < num; x1++){
    		answer[x1] = kari[x1][18] * 0.01470588235;
    		System.out.println(answer[x1]);
		}
    	
    	for(x1 = 0; x1 < num; x1++){
    		for(x2 = 0; x2 < n-1; x2++){
    			x[x1][x2] = kari[x1][x2];
    		}
    	}
    	
         
        // パーセプトロンの動作確認
        try
        {
            // 標準出力をファイルに関連付ける
            String      fileName    = System.getProperty( "user.dir" )
                                      + "/"
                                      + "TestMultiLayerPerceptron.log";
            PrintStream out         = new PrintStream( fileName );
            System.setOut( out );
             
            // 多層パーセプトロンの作成
            MultiLayerPerceptron    mlp = new MultiLayerPerceptron( 18 , 18 , 1 );
            mlp.learn( x , answer );
             
            // ファイルを閉じる
            out.close();
             
        }catch( Exception e ){
            e.printStackTrace();
        }
    }

}

class MultiLayerPerceptron
{
    // 定数
    protected static final int      MAX_TRIAL   = 1000000000;       // 最大試行回数
    protected static final double   MAX_GAP     = 0.001;          // 出力値で許容する誤差の最大値
     
    // プロパティ
    protected int   inputNumber     = 0;
    protected int   middleNumber    = 0;
    protected int   outputNumber    = 0;
    protected Neuron[]    middleNeurons   = null;     // 中間層のニューロン
    protected Neuron[]    outputNeurons   = null;     // 出力層のニューロン
     
    // ロガー
    protected Logger      logger    = Logger.getAnonymousLogger();  // ログ出力
     
    /**
     * 三層パーセプトロンの初期化
     * @param input  入力層のニューロン数
     * @param middle 中間層のニューロン数
     * @param output 出力層のニューロン数
     */
    public MultiLayerPerceptron( int input , int middle , int output )
    {
        // 内部変数の初期化
        this.inputNumber    = input;
        this.middleNumber   = middle;
        this.outputNumber   = output;
        this.middleNeurons  = new Neuron[middle];
        this.outputNeurons  = new Neuron[output];

        // 中間層のニューロン作成
        //flag
        //for( int i=0 ; i<middle ; i++ ){ middleNeurons[i] = new Neuron( input ); }
        for( int i=0 ; i<middle ; i++ ){ middleNeurons[i] = new Neuron( middle ); }
         
        // 出力層のニューロン作成
        //for( int i=0 ; i<output ; i++ ){ outputNeurons[i] = new Neuron( input ); }
        for( int i=0 ; i<output ; i++ ){ outputNeurons[i] = new Neuron( middle ); }
         
        // 確認メッセージ
        System.out.println( "[init]  " + this );
    }
     
    /**
     * 学習
     * @param x
     * @param answer
     */
    public void learn( double[][] x , double[] answer )
    {
        // 変数初期化
        double[]    in      = null;                           // i回目の試行で利用する教師入力データ
        double      ans     = 0;                              // i回目の試行で利用する教師出力データ
        double[]    h       = new double[ middleNumber ];     // 中間層の出力
        double[]    o       = new double[ outputNumber ];     // 出力層の出力
        int i = 0;
        double[][] seikai = new double  [500][1];
        double seigo = 0;
         
        // 学習
        int succeed = 0;        // 連続正解回数を初期化
        for(  i=0 ; i<MAX_TRIAL ; i++ )
        {
            // 行間を空ける
        	if(i % 10000000  == 0){
            System.out.println();
            System.out.println( String.format( "Trial:%d" , i ) );
        	}
            // 使用する教師データを選択
            in  = x[ i % answer.length ];
            ans = answer[ i % answer.length ];
             
            // 出力値を推定：中間層の出力計算
            /*if(i == 0){
            	 h = new double []{-31.823227990187966 , -28.844724800949734 , 15.272144176167343 , 5.891914407024041 , 19.88447075451847 , -6.797151445245121 , 19.197306335555595 , 15.856340602389912 , 11.439134545562592 , -2.8824648214126944 , 15.33888737215773 , -7.857715891487612 , 10.044164840803763 , -3.58897790386485 , -23.329400697187133 , 13.981672867211497 , -7.448495945316647 , 27.030649936900378};
            }else{
            for( int j=0 ; j<middleNumber ; j++ )
            {
                h[j] = middleNeurons[j].output( in );
            }
            }*/
            for( int j=0 ; j<middleNumber ; j++ )
            {
                h[j] = middleNeurons[j].output( in );
            }
             
            // 出力値を推定：出力層の出力計算
            for( int j=0 ; j<outputNumber ; j++ )
            {
                o[j] = outputNeurons[j].output( h );
            }
            if(i % 10000000  == 0){
            System.out.println( String.format( "[input] %f , %f , %f" , in[0] , in[1] , in[2] ) );
            System.out.println( String.format( "[answer] %f" , ans ) );
            System.out.println( String.format( "[output] %f" , o[0] ) );
            System.out.println( String.format( "[middle] %f",h[0]) );
            System.out.println(Math.abs(ans - o[0]));
            }
            if(Math.abs(ans - o[0]) < 0.01470588235){
            	seikai[i % 500][0] = 1;
            }
            // 評価・判定
            boolean successFlg  = true;
            for( int j=0 ; j<outputNumber ; j++ )
            {
                // 出力層ニューロンの学習定数δを計算
                double delta = ( ans - o[j] ) * o[j] * ( 1.0d - o[j] );
                 
                // 教師データとの誤差が十分小さい場合は次の処理へ
                // そうでなければ正解フラグを初期化
                if( Math.abs( ans - o[j] ) < MAX_GAP ){ continue; }
                                                  else{ successFlg = false; }
                 
                // 学習
                //if(i % 1000 == 0){
                //System.out.println( "[learn] before o :" + outputNeurons[j] );
                //}
                outputNeurons[j].learn( delta , h );
                //if(i % 10000 == 0){
                //System.out.println( "[learn] after o  :" + outputNeurons[j] );
                //}
                 
            }
             
            // 連続成功回数による終了判定
            if( successFlg )
            {
                // 連続成功回数をインクリメントして、
                // 終了条件を満たすか確認
                succeed++;
                if( succeed >= x.length ){ break; }else{ continue; }
            }else{
                succeed = 0;
            }
             
            // 中間層の更新
            for( int j=0 ; j<middleNumber ; j++ )
            {   
                // 中間層ニューロンの学習定数δを計算
                double sumDelta = 0;
                for( int k=0 ; k<outputNumber ; k++ )
                {
                    Neuron  n    = outputNeurons[k];
                    sumDelta    += n.getInputWeightIndexOf(j) * n.getDelta(); 
                }
                double delta = h[j] * ( 1.0d - h[j] ) * sumDelta;
                 
                // 学習
                //if(i % 1000 == 0){
                //System.out.println( "[learn] before m :" + middleNeurons[j] );
                //}
                middleNeurons[j].learn( delta , in );
                //if(i % 1000 == 0){
                //System.out.println( "[learn] after m  :" + middleNeurons[j] );
                //}
            }
 
             
            // 再度出力
            // 出力値を推定：中間層の出力計算
            //h[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
            
            for( int j=0 ; j<middleNumber ; j++ )
            {
                h[j] = middleNeurons[j].output( in );
            }
            
             
            /*for( int j=0 ; j<middleNumber ; j++ )
            {
                h[j] = middleNeurons[j].output( in );
            }*/
            // 出力値を推定：出力層の出力計算
            
            for( int j=0 ; j<outputNumber ; j++ )
            {
                o[j] = outputNeurons[j].output( h );
            }
              
            if(i % 10000000  == 0){
            System.out.println( String.format( "[input] %f , %f" , in[0] , in[1] ) );
            System.out.println( String.format( "[answer] %f" , ans ));
            System.out.println( String.format( "[output] %f" , o[0]) );
            System.out.println( String.format( "[middle] %f , "
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f"
            		+ "%f" , h[0] , h[1] ,h[2],h[3],h[4],h[5],h[6],h[7],h[8],h[9],h[10],h[11],h[12],h[13],h[14],h[15],h[16],h[17],h[17]) );
            }
            if(Math.abs(ans - o[0]) < 0.01470588235){
            	seikai[i % 500][0] = 1;
            }
        }
         
        // すべての教師データで正解を出すか
        // 収束限度回数を超えた場合に終了
        System.out.println( "[finish] " + this );
        System.out.println( String.format( "Trial:%d" , i ) );
        //正誤判定
        for(int j = 0; j < 500; j++){
        	System.out.println(seikai[j][0]);
        	seigo += seikai[j][0];
        }
        System.out.println( String.format("正解回数は %f" , seigo ));
    }
     
    @Override
    public String toString()
    {
        // 戻り値変数
        String str  = "";
         
        // 中間層ニューロン出力
        str += " middle neurons ( ";
        for( Neuron n : middleNeurons ){ str += n; }
        str += ") ";
         
        // 出力層ニューロン出力
        str += " output neurons ( ";
        for( Neuron n : outputNeurons ){ str += n; }
        str += ") ";
         
        return str;
    }
     
     
     
    /**
     * 多層パーセプトロン内部で利用するニューロン 
     * @author karura
     */
    class Neuron
    {
         
        // 内部変数
        protected int         inputNeuronNum  = 0;         // 入力の数
        protected double[]    inputWeights    = null;      // 入力ごとの結合加重
        protected double      delta           = 0;         // 学習定数δ
        protected double      threshold       = 0;         // 閾値θ
        protected double      eater           = 1.0d;      // 学習係数η
         
        /**
         * 初期化
         * @param inputNeuronNum 入力ニューロン数
         */
        public Neuron( int inputNeuronNum )
        {
            // 変数初期化
            Random r = new Random();
            this.inputNeuronNum = inputNeuronNum;
            this.inputWeights   = new double[ inputNeuronNum ];
            this.threshold      = r.nextDouble();               // 閾値をランダムに生成
             
            // 結合加重を乱数で初期化
            for( int i=0 ; i<inputWeights.length ; i++ )
            { this.inputWeights[i] = r.nextDouble();}
        }
         
        /**
         * 学習（バックプロパゲーション学習）
         * @param inputValues 入力データ
         * @param delta       δ
         */
        public void learn( double delta , double[] inputValues  )
        {
            // 内部変数の更新
            this.delta  = delta;
             
            // 結合加重の更新
            for( int i=0 ; i<inputWeights.length ; i++ )
            {
                // バックプロパゲーション学習
                inputWeights[i] += eater * delta * inputValues[i];
            }
             
            // 閾値の更新
            threshold -= eater * delta; 
        }
 
        /**
         * 計算
         * @param  inputValues 入力ニューロンからの入力値
         * @return 推定値
         */
        public double output( double[] inputValues )
        {
            // 入力値の総和を計算
            double  sum = -threshold;
            for( int i=0 ; i<inputValues.length ; i++ )
            { sum += inputValues[i] * inputWeights[i]; }
             
            // 活性化関数を適用して、出力値を計算
            double  out = activation( sum );
             
            return out;
        }
         
        /**
         * 活性化関数（シグモイド関数）
         * @param x
         * @return
         */
        protected double activation( double x )
        {
            return 1 / ( 1 + Math.pow( Math.E , -x ) );
        }
         
        /**
         * 
         * 入力iに対する結合加重を取得
         * @param i
         * @return
         */
        public double getInputWeightIndexOf( int i )
        {
            if( i>=inputNumber ){ new RuntimeException("outbound of index"); }
            return inputWeights[i];
        }
 
        /**
         * 学習定数δの取得
         * @return 学習定数δ
         */
        public double getDelta()
        {
            return delta;
        }
 
        /**
         * クラス内部確認用の文字列出力
         */
        @Override
        public String toString()
        {
            // 出力文字列の作成
            String output = "weight : ";
            for( int i=0 ; i<inputNeuronNum ; i++ ){ output += inputWeights[i] + " , "; }
             
            return output;
             
        }
         
    }
 
}
