from openai import OpenAI
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import pandas as pd
import base64

client = OpenAI(api_key="")
MODEL = 'gpt-4o'

with open("image/Slide16.png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

def validate_data(input_data):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""
You are a helpful assistant designed to predict whether a driver will yield to a pedestrian at unsignalized intersections. You will be provided with a line of interaction data between the driver’s vehicle and the pedestrian, along with a photo of the intersection. Your task is to predict whether the driver yields, based on the data, the intersection photo, and the following relevant prompts.
- Consider the effect of each variable on the relinquishing behavior
- Focus solely on the information provided without making assumptions beyond the given data.
- Each column in the data is explained in the following order:
• Time Showed Intent: the time that the pedestrian first arrives at the intersections and indicates intent to cross presented in hour, minute, second format (e.g., 193825 for 7:38:26pm)
• Time Started Crossing: the time that the pedestrian begins the crossing presented in hour, minute, second format (e.g., 193825 for 7:38:26pm)
• Number of Pedestrians: the number of pedestrians in the party
• Pedestrian Origin: letter A-D indicating which intersection corner the pedestrian originated at – see Deliverable 6 for full details
• Pedestrian Destination: letter A-D indicating which intersection corner the pedestrian ends at – see Deliverable 6 for full details
• Pedestrian type: A: person on foot; B: person on bike; C: person on vehicle (e.g., scooter/hoverboard); D: person walking bike; E: mix of pedestrian types; F: other (see comments for entry); G: person with a dog; H: person with a stroller or small child
• Vehicle Speed: estimate of vehicle speed (mph)
• Opposite Direction Yield: 0 if no opposite direction vehicle, 1 if the vehicle traveling in the opposite direction yields, 2 if the vehicle traveling in the opposite direction does not yield
• Following Vehicle: 1 if there was a vehicle behind the interaction vehicle, 0 if not
• Posted Speed: Posted speed limit (mph)
• Number of Lanes at Main Street: number of lanes that the pedestrian is crossing
• Crossing Width (Major): pedestrian crossing distance (ft)
• Bike Lane(s): 0 if no bike lanes present on the major road, 1 if there are bike lanes present
• Weather: 0 if there is no precipitation; 1 if it is raining; 2 if it is snowing
• Signage: 0 if there are no signs for the crosswalk, 1 if the crosswalk is signed
• Markings: U for unmarked, C for continental (zebra) markings, S for standard (two solid white lines)
• Presence of Single Family: adjacent land use indicating presence of single family homes within a 1 block radius of the intersection, 1 for present, 0 for not present
• Presence of Apartments: adjacent land use indicating presence of multifamily homes within a 1 block radius of the intersection, 1 for present, 0 for not present
• Presence of Commercial: adjacent land use indicating presence of commercial buildings within a 1 block radius of the intersection, 1 for present, 0 for not present
• Presence of Gas Station/Convenient Store: adjacent land use indicating presence of a gas station or convenience store within a 1 block radius of the intersection, 1 for present, 0 for not present
• Presence of Restaurants/Bars: adjacent land use indicating presence of bars or restaurants within a 1 block radius of the intersection, 1 for present, 0 for not present
• Presence of Parking Lots: adjacent land use indicating presence of large street-facing parking lots (not on-street parking) within a 1 block radius of the intersection, 1 for present, 0 for not present
• Lighting: 0 for no lighting, 1 for lighting
• Road surface: 0 for dry, 1 for wet
• Number of bus stops near the intersection: number of bus stops within one block of the intersection on the main road
• Minor AADT: annual average daily traffic on the minor road
• Major AADT: annual average daily traffic on the major road
• Dist. to Nearest Park: distance in miles to nearest park
• Dist. to Nearest School: distance in miles to nearest school
• Presence of on street parking: presence of on-street parking, 0 – no parking; 1 – one- sided parking; 2 – two-sided parking
• PAWS Score: PAWS score
• Tree Cover: 0-4 indicating the number of crossing points covered by trees Additionally, the following are identified for each interaction for vehicles traveling in both for Direction 1 (vehicles traveling toward the camera) and Direction 2 (vehicles traveling away from the camera), Note that in the data, Direction 1 is in the front and Direction 2 is in the back:
Note: cells that are left empty are either not applicable (e.g., no vehicle to report) or were not collected for that interaction technical reasons. Often, this is because a sight obstruction made it difficult to accurately assess.

- Relevant domain knowledge:
• The influence of attributes is ranked in the following order: Vehicle speed, Crossing width (major), Opposite direction yield, Presence of parking lots, Presence of Restaurants/Bars, Distance to the nearest park, and Distance to the nearest school. These attributes mainly affect the driver's behavior of yield.
• Vehicle Dynamics and Control
• Vehicle Dynamics and Control
    1. Vehicle speed is the most influential factor affecting driver yielding behavior.
    2. As vehicle speed increases, the probability of yielding decreases significantly.
    3. When speed is below 10 MPH, drivers are very likely to yield.
    4. When speed is between 10–20 MPH, yielding becomes uncertain (approximately equal chance).
    5. When speed exceeds 20 MPH, drivers are very unlikely to yield.
    6. Roads with a 30 MPH speed limit have a higher yielding rate than those with 35 MPH.
    7. Opposite-direction yielding has a strong social influence:
        -If the opposite vehicle yields, the subject driver is very likely to yield.
        -If the opposite vehicle does not yield, the subject driver is very unlikely to yield.

• Road networks and infrastructure
    1. Road Geometry and Traffic Environment
        -Wider crossing width on the major road is associated with a higher probability of driver yielding.
        -More traffic lanes on the major road reduce the likelihood of yielding.
        -A higher number of pedestrians near the intersection increases the probability of yielding.
        -A greater number of nearby bus stops is associated with higher yielding rates.
    2. Built Environment Context
        -The presence of restaurants/bars nearby greatly increases the likelihood of driver yielding.
        -The presence of parking lots generally reduces the likelihood of yielding.
        -The presence of commercial buildings, gas stations, and apartments is associated with higher yielding probability.
        -The presence of single-family housing is associated with higher yielding probability.
        -The presence of on-street parking on both sides is associated with higher yielding rates than no parking.
    3. Surrounding Sensitive Land Uses
        -A shorter distance to a park strongly increases the probability of driver yielding.
        -A shorter distance to a school also increases the probability of yielding.
    4. Traffic Control and Facilities
        -Standard pedestrian crosswalk markings produce the highest yielding rates.
        -Unmarked crosswalks have the lowest yielding rates.
        -The presence of traffic signage significantly increases the likelihood of yielding.
        -The presence of bike lanes is associated with a lower probability of yielding.

• Pedestrian Mobility and Interaction Assessment.
    1. Drivers are more likely to yield to groups of pedestrians than to a single pedestrian.
    2. Pedestrians with strollers or children significantly increase the likelihood of yielding.
    3. Pedestrians walking with a dog moderately increase yielding probability.
    4. Pedestrians using bikes or vehicles are associated with a lower probability of yielding.
    5. Mixed pedestrian groups generally receive higher yielding rates than single-mode users.

-Here are some steps to guide you on how to think:
Step 1: Analyze vehicle attributes.Through relevant knowledge and one's own reasoning.
Step 2: Evaluate road conditions.Through relevant knowledge, pictures and one's own reasoning.
Step 3: Analyze pedestrian-related characteristics.Through relevant knowledge and one's own reasoning.
Step 4: Establish the priority of factors influencing driver yielding behavior. Prioritize the following features: Vehicle speed, Crossing width (major), Opposite direction yield, Presence of parking lots, Presence of Restaurants/Bars, Distance to the nearest park, and Distance to the nearest school. Then combine the other characteristics to get the final result, and the reason for the result.
Step 5: Result and reson.

-Here are some examples:
Example 1:
DATA:   "Time Showed Intent (Axis_Foxtrot)": "201046",
        "Time Started Crossing (Foxtrot)": "201047",
        "Number of Pedestrians": "1",
        "Pedestrian Origin": "A",
        "Pedestrian Destination": "B",
        "Pedestrian Type": "A",
        "Vehicle Speed": "13.6",
        "Opposite Direction Yield": "0",
        "Following Vehicle": "1",
        "Posted Speed": "35",
        "Number of Lanes at Main Street": "4",
        "Crossing Width (Major)": "63",
        "Bike Lane(s)": "0",
        "Weather": "0",
        "Signage": "0",
        "Markings": "U",
        "Presence of Single Family": "1",
        "Presence of Apartments ": "1",
        "Presence of Commercial": "1",
        "Presence of Gas Station/ Convenient Store": "0",
        "Presence of Restaurants/ Bars": "1",
        "Presence of Parking Lots": "1",
        "Dist. to Nearest Park": "0.53",
        "Dist. to Nearest School": "4.4",
        "Presence of on street parking ": "0",
        "PAWS Score": "12",
        "Tree Cover ": "1",
        "lighting": "0",
        "road surface": "0",
        "# of bus stops near the intersection": "0",
        "Minor AADT": NaN,
        "Major AADT": "10500"
        
Step 1: Vehicle Dynamics and Control
        "Vehicle Speed": 13.6 → Above low-speed threshold (10 MPH) → Lower likelihood of yielding.
        "Posted Speed": 35 → High-speed road, likely multi-lane, reduces driver attention to pedestrians.
        Overall: Speed is the main influencing factor in vehicle attributes, but 13.6MPH is not in the low speed range or the high speed range, so it cannot be judged.  The yield rate is reduced from the perspective of comprehensive vehicle factors.

Step 2: Road networks and infrastructure
        "Opposite Direction Yield": 0 → no opposite direction vehicle
        "Crossing Width (Major)": 63 → Wide, multi-lane crossing → harder for pedestrians to cross, lower yielding likelihood.
        "Presence of Restaurants/ Bars": 1, "Presence of Parking Lots": 1 → Commercial area → slightly increases pedestrian visibility 
        "Dist. to Nearest Park": "0.53","Dist. to Nearest School": "4.4",
        Overall: Crossing Width (Major), Presence of Restaurants/Bars, Dist. to nearest school, Presence of parking lots and Dist. to nearest park are the main influencing factors in road attributes. Although the road width is relatively wide and there are Restaurants/Bars, there are parking lots, and the road is relatively close to the school and park. Judge to slightly reduce the yield rate.

Step 3: Pedestrian Mobility and Interaction Assessment
        "Pedestrian Type": "A" → Person on foot 
        "Interaction/ Event Type": "A" → Pedestrian crossed at a comfortable pace 
        "Number of Pedestrians": 1 → Only one pedestrian; simpler situation.
        Overall:Pedestrians cross the road at a comfortable pace, reducing the yield rate

Step 4: "Time Started Crossing (Foxtrot)": "201047",
        "Weather" = 0 → Clear weather, no rain or snow → normal visibility 
        "Signage": 0 → No crosswalk signage 
        "Markings": U → Unmarked crosswalk 
        Overall: At night, there are no signs and no sidewalks, and the weather is not rainy, reducing the yield rate.

Step 5: result: False, reson:From the perspective of main factors, it is impossible to judge the impact of speed and non-yielding of object vehicles on yielding. Although the road is wider and there are parking lots, these factors increase the yield rate, but the proximity to schools and parks, and the existence of parking Spaces reduce the yield rate. Considering pedestrian factors and other factors, it is judged that the driver will not yield.

As a simple example, the above requires to analyze the impact of the provided data on the driver's behavior according to the prompts of relevant field information and your reasoning of various attributes, and deduce the results and reasons according to the provided thinking guidance steps.

**Return only a JSON object** with the following two properties:
- `"pred_result"`: a boolean (`true` or `false`) Indicates whether the driver will yield
- `"pred_reson"`: Provide a reason for predicting the driver’s yielding behavior, limited to 150 words.

Both JSON properties must always be present.

Do not include any additional text or explanations outside the JSON object.
DATA:
{input_data}
                """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
    }
]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    response_content = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
    
    try:
        if isinstance(response_content, dict):
            response_dict = response_content
        else:
            response_dict = json.loads(response_content)
        return response_dict
    except json.JSONDecodeError as e:
        #print(f"Failed to decode JSON response: {response_content}")
        raise e

#Read the CSV file and exclude the label columns
#input_data = pd.read_excel('data.xlsx')
input_data = pd.read_csv('split_by_location_opposite/location_16.csv')
input_data = input_data.iloc[:, :-1]
input_data = input_data.values.tolist()



# Function to validate a single row of data
def validate_row(row):
    input_str = ','.join(map(str, row))
    result_json = validate_data(input_str)
    return result_json



# Validate data rows and collect results
pred_result = [False] * len(input_data)
pred_reson = [''] * len(input_data)

for i, row in enumerate(input_data):
    result_json = validate_row(row)  # 调用 validate_row 函数，获得结果
    #print(result_json)  # 输出每次验证的结果
    pred_result[i] = result_json['pred_result']  # 将预测结果存储在 pred_result 中
    pred_reson[i] = result_json['pred_reson']  

#test the results
#print(pred_result)
#print(pred_reson)

df = pd.DataFrame({
    'result': pred_result,
    'reson': pred_reson
})
df.to_csv('new_knowledge/output16.csv', index=False, encoding='utf-8')