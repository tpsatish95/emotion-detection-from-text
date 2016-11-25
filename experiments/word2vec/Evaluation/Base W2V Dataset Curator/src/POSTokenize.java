import java.io.*;
import java.util.*;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
public class POSTokenize {

	public static void main(String args[])
	{
		List<String> a = null;
		int emoNUM=0;
		FileInputStream stop=null;
		FileInputStream emo[]= new FileInputStream[6];
		FileOutputStream emoOUT[] = new FileOutputStream[6];
		MaxentTagger tagger=null;
		try {
			tagger = new MaxentTagger("models/left3words-wsj-0-18.tagger");
		} catch (ClassNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		
		List<TaggedWord> wPOS ;
		
		
		String punc[]={"1","2","3","4","5","6","7","8","9","0",",",":","(",")","[","]",".","-","!","?","\""};
		
		try {
			

		emo[0] = new FileInputStream("joy.txt");
		 emo[1] = new FileInputStream("anger.txt");
		 emo[2] = new FileInputStream("disgust.txt");
		 emo[3] = new FileInputStream("sadness.txt");
		 emo[4] = new FileInputStream("surprise.txt");
		 emo[5] = new FileInputStream("fear.txt");
		 
		 emoOUT[0] = new FileOutputStream("joyO.txt");
		 emoOUT[1] = new FileOutputStream("angerO.txt");
		 emoOUT[2] = new FileOutputStream("disgustO.txt");
		 emoOUT[3] = new FileOutputStream("sadnessO.txt");
		 emoOUT[4] = new FileOutputStream("surpriseO.txt");
		 emoOUT[5] = new FileOutputStream("fearO.txt");
		 
		 stop = new FileInputStream("stop.txt");
		 
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			System.out.println("File Not Found");
		}
		String emotions[] = {"joy","anger","disgust","sadness","surprise","fear"};
		
		String negs[]={"cannot","do not","fail","little value","mistake","not","problem","unable to","unfortunately","don't","can't"};
		
		String ImpTags[]={"JJ","JJR","JJS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"};
		
		String sp[]= new String[174];
		int o=0;
		Scanner sin = new Scanner(stop);
		while(sin.hasNextLine())
		{				
				sp[o++] = sin.nextLine();
				
		}
		System.out.println("Stop WOrds Over!!");
		
		while(emoNUM<=5)
		{
		
			Scanner in = new Scanner(emo[emoNUM]);
			
			String temp;
			
			while(in.hasNextLine())
			{
				
				try
				{
				
				temp= in.nextLine();
				//System.out.println("!");
		
				String lines[]=temp.split("[.][ ]");
				
				for (String line : lines)
				{	
				
				StringReader sr = new StringReader(line);
				
				List<List<HasWord>> wL = tagger.tokenizeText(sr);
				
			
				wPOS = tagger.tagSentence(wL.get(0));
				
				for(int t=0;t<wPOS.size();t++)
				{
					for(int h=0;h<ImpTags.length;h++)
					{
						if(wPOS.get(t).toString().contains(ImpTags[h]))
					{
					
							// because of dataset :(
							if(wPOS.get(t).word().toLowerCase().trim().contains("surpris") && emoNUM==4)		
							{
							break;	
							}
							if(wPOS.get(t).word().toLowerCase().trim().contains("rrb")||wPOS.get(t).word().toLowerCase().trim().contains("lrb") || wPOS.get(t).word().toLowerCase().trim().contains(" "))		
							{
							break;	
							}
					emoOUT[emoNUM].write(wPOS.get(t).word().getBytes());
					emoOUT[emoNUM].write(" ".getBytes());
					
					break;
					
					}
				}
				}
				
				/*int flag=0;
				if(emoNUM==4)
					for(int j=0;j<negs.length;j++)
						if(temp.contains(negs[j]))
							{flag=1;break;}
				
				
				if(flag==1)
					continue;
				
				
				//Twokenize
				try {
					 a= twokenize.retTOK(temp);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				//remove stop words
				for (int i=0; i<a.size(); i++) {
	    		int f=0,f1=0;
					String tt =	a.get(i);
	    		
	    		for(int k=0;k<sp.length;k++)
	    		{
	    			if(sp[k].compareToIgnoreCase(tt)==0)
	    			{
	    				f=1;
	    				break;
	    			}
	    		}
	    		
	    		if(f==1)
	    		{a.remove(i);	continue;}
	    		
	    		for(int y=0;y<punc.length;y++)
	    		{
	    			if(tt.contains(punc[y]))	
		    		{f1=1; break;}	
	    		}
	    		
	    		if(f1==1)
	    		{a.remove(i);	continue;}
	    		
	    	    if(emoNUM==4)
	    	    	if(tt.toLowerCase().contains("surpris"))
	    	    	{a.remove(i);continue;}
	    		
                // Feed in to word2Vec in C
				
				// TOo time consuming soooooo just store output in file
				
				}
				*/
				/*for(int i=0;i<a.size();i++)
				{
					emoOUT[emoNUM].write(a.get(i).getBytes());
					emoOUT[emoNUM].write(" ".getBytes());
				}
				*/
				
							
				}
				emoOUT[emoNUM].write("\n".getBytes());		
				}//try
				catch(Exception e)
				{
					System.out.println("End of FILE"+emoNUM);
					e.printStackTrace();
				}
			}
			try {
				emoOUT[emoNUM].flush();
				emoOUT[emoNUM].close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			emoNUM++;
		}//while
	}
}
