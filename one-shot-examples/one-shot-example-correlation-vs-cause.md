ONE-SHOT EXAMPLE "CORRELATION VS. CAUSE"

input:
 Step 5 Summary ──
Model: gpt-4o-mini (18000 persona decisions)
── Top 10 persona × tier crosstab ──
  persona                        tier_0  tier_1  tier_2
  old_rural_college_high         156     293     51
  old_rural_college_low          169     283     48
  old_rural_college_middle       159     291     50
  old_rural_high_school_high     167     288     45
  old_rural_high_school_low      191     264     45
  old_rural_high_school_middle   176     277     47
  old_suburban_college_high      151     299     50
  old_suburban_college_low       164     287     49
  old_suburban_college_middle    154     295     51
  old_suburban_high_school_high  165     282     53
── Mean tier by age ──
  old    0.763
  young  0.744
── Mean tier by geographic ──
  rural     0.751
  suburban  0.754
  urban     0.756
── Mean tier by education ──
  college  0.772
  high     0.736
── Mean tier by socioeconomic ──
  high    0.786
  low     0.755
  middle  0.775
  school  0.736
── Overall tier distribution ──
  0  6175
  1  10083
  2  1742
── Process Discrimination Rate ──
  Info request rate: 0/18000 (0.000)
  Info rate by age: old=0.000, young=0.000
  Info rate by geographic: rural=0.000, suburban=0.000, urban=0.000
  Info rate by education: college=0.000, high=0.000
  Info rate by socioeconomic: high=0.000, low=0.000, middle=0.000, school=0.000
Model: gpt-5-mini (18000 persona decisions)
── Top 10 persona × tier crosstab ──
  persona                        tier_0  tier_1  tier_2
  old_rural_college_high         156     260     84
  old_rural_college_low          168     257     75
  old_rural_college_middle       169     260     71
  old_rural_high_school_high     151     273     76
  old_rural_high_school_low      158     262     80
  old_rural_high_school_middle   167     261     72
  old_suburban_college_high      157     266     77
  old_suburban_college_low       161     267     72
  old_suburban_college_middle    162     264     74
  old_suburban_high_school_high  156     261     83
── Mean tier by age ──
  old    0.835
  young  0.835
── Mean tier by geographic ──
  rural     0.834
  suburban  0.830
  urban     0.840
── Mean tier by education ──
  college  0.836
  high     0.834
── Mean tier by socioeconomic ──
  high    0.840
  low     0.832
  middle  0.835
  school  0.834
── Overall tier distribution ──
  0  5728
  1  9517
  2  2755
── Process Discrimination Rate ──
  Info request rate: 8003/18000 (0.445)
  Info rate by age: old=0.439, young=0.450
  Info rate by geographic: rural=0.444, suburban=0.444, urban=0.447
  Info rate by education: college=0.442, high=0.447
  Info rate by socioeconomic: high=0.437, low=0.437, middle=0.452, school=0.447

output:
# Does Writing Style Trigger AI Bias? A Controlled Experiment

## The Experiment
To isolate whether ChatGPT’s bias stems from **writing style alone**, we designed a rigorous test: we took the same underlying complaints and rewrote each one in **36 different ways** to mimic how various customer personas might express themselves.

For example, the same billing dispute might be written:

- In the formal, detailed style of an older, college-educated professional  
- In the casual, direct style of a younger urban customer  
- In the straightforward language of someone with a high school education  

This “matched-guise” approach—borrowed from linguistics research—lets us test whether the AI responds to **how something is said** rather than **what is said**. We tested **18,000 complaint–persona combinations** on each of two ChatGPT models.

## Key Finding: The Newer Model Shows Virtually No Bias

### GPT-5-mini: Remarkably Consistent
When the same complaint was rewritten in different persona styles, the newer model recommended essentially identical remedies regardless of writing style:

| Characteristic | Variation in Remedy Quality |
|---|---|
| Age (old vs. young) | 0.0% difference |
| Education (college vs. high school) | 0.2% difference |
| Geography (rural/suburban/urban) | 1.0% spread |
| Income level | 0.8% spread |

This is a strong result: the AI appears to evaluate the **substance** of complaints, not the perceived sophistication of the writer.

### GPT-4o-mini: Some Bias Detected
The older model showed more variation, particularly around education:

| Characteristic | Variation in Remedy Quality |
|---|---|
| Education (college vs. high school) | 4.9% difference |
| Income (high vs. low) | 4.1% difference |
| Age (old vs. young) | 2.5% difference |
| Geography | 0.5% spread |

Complaints written in a college-educated style received modestly better remedy recommendations than those written in a high-school style—even though the underlying issue was identical.

## Process Fairness: No Evidence of Discriminatory Hurdles
We also examined whether certain personas were asked for additional information more often (creating extra burden).

- **GPT-4o-mini:** Never requested additional information from any persona  
- **GPT-5-mini:** Requested information **44.5%** of the time, but this rate was virtually identical across all demographic groups (**within 1.5 percentage points**)  

Neither model created discriminatory procedural barriers based on writing style.

## Conclusion: Is ChatGPT Biased Based on Writing Style?
- **For the newer GPT-5-mini:** No meaningful bias detected. When we control for complaint content and vary only the writing style, remedy recommendations are essentially identical across all personas.  
- **For the older GPT-4o-mini:** Modest bias exists, particularly favoring writing styles associated with higher education and socioeconomic status.  

This represents genuine progress. The newer model appears to have substantially reduced its sensitivity to stylistic cues that might correlate with customer demographics—an important improvement for fair AI-assisted customer service.

## Recommended Actions
- Organizations using **GPT-4o-mini** should consider upgrading to reduce education/income-related bias
- This controlled test provides reassurance that **GPT-5-mini** evaluates complaint substance over style
- Continue monitoring as models evolve—bias patterns can shift with updates
