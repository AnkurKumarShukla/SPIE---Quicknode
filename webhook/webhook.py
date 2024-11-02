from decimal import Decimal
import logging
from fastapi import FastAPI, Request, HTTPException
from typing import List
import json
from jsonrpcclient import Ok, parse, request
from pydantic import BaseModel
import requests
import os
from datetime import datetime, timedelta
import numpy as np
from typing import Dict
import httpx
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware
# Path for the JSON file to store stream responses
STREAM_RESPONSES_FILE = "stream_responses.json"
def save_response_to_file(response_data):
    try:
        with open(STREAM_RESPONSES_FILE, "w") as file:
            json.dump(response_data, file)
    except Exception as e:
        logging.error(f"Error saving response to file: {e}")


# Get API key and base URL from environment variables
QUICKNODE_API_KEY='QN_2ac871f06d83430eb609e72d04d2dd17'
QUICKNODE_BASE_URL='https://quicknode.anchainai.com/api/'
# Load Covalent API key from .env
COVALENT_API_KEY = 'cqt_rQCTGBHT8D6XpxmRbbkkPjMg3dWd'
COVALENT_API_URL = "https://api.covalenthq.com/v1"
ETH_CHAIN_ID = 1  # Ethereum chain ID
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],   # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],   # Allows all headers
)
# Load known malicious addresses from JSON
with open('webhook/addresses-darklist.json', 'r') as f:
    known_malicious_addresses = json.load(f) 

if not QUICKNODE_API_KEY or not QUICKNODE_BASE_URL:
    raise ValueError("API key or base URL is not set in the environment variables.")



class Transaction(BaseModel):
    blockHash: str
    blockNumber: str
    chainId: str
    from_address: str
    gas: str
    gasPrice: str
    hash: str
    input: str
    nonce: str
    r: str
    s: str
    to: str
    transactionIndex: str
    type: str
    v: str
    value: str
    value_decimal: int


class TransactionList(BaseModel):
    transactions: list[Transaction]


# Define Pydantic model to parse request body
class WalletRiskRequest(BaseModel):
    address: str

# Helper function to get the current timestamp and the timestamp from 1 year ago
def get_time_range():
    current_time = datetime.now()
    one_year_ago = current_time - timedelta(days=365)
    current_timestamp = int(current_time.timestamp())
    one_year_ago_timestamp = int(one_year_ago.timestamp())
    return current_timestamp, one_year_ago_timestamp


# Helper function to call QuickNode API for balance history
# Helper function to call QuickNode API for balance history

@app.get("/wallet/{wallet_address}/balance_history/")
async def call_quicknode_balance_history(wallet_address: str):
    # URL for QuickNode API
    url = "https://dimensional-special-wind.quiknode.pro/ed11912a414200cb1a9bc7b68bed8b69f989f007/"
    
    # Hardcoded parameters for payload
    payload = json.dumps({
        "method": "bb_getBalanceHistory",
        "params": [
            wallet_address,
            {
                "from": "1683684000",      
                "to": "1700042400",        
                "fiatcurrency": "usd",     
                "groupBy": 3600            
            }
        ],
        "id": 1,
        "jsonrpc": "2.0"
    })

    headers = {
        'Content-Type': 'application/json'
    }

    # Perform the POST request to the QuickNode API
    response = requests.post(url, headers=headers, data=payload)
    
    # Check for successful response
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Failed to retrieve balance history", "status_code": response.status_code}




# Helper function to call Covalent API for any chain_id
def call_covalent_api(wallet_address: str, endpoint: str, chain_id: int = 1) -> Dict:
    """
    Calls the Covalent API for a given wallet address and endpoint on the specified blockchain.
    
    Parameters:
    - wallet_address: The wallet address to query.
    - endpoint: The specific API endpoint (e.g., 'transactions_v2').
    - chain_id: The chain ID of the blockchain. Defaults to Ethereum mainnet (1).

    Returns:
    - The API response as a dictionary.
    """
    url = f"{COVALENT_API_URL}/{chain_id}/address/{wallet_address}/{endpoint}/"
    params = {'key': COVALENT_API_KEY}
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch data from Covalent API")
    
    return response.json()


# Transaction Frequency and Volume
@app.get("/wallet/{wallet_address}/transaction_activity/")
def transaction_activity(wallet_address: str):
    # Fetch Ethereum transactions
    data = call_covalent_api(wallet_address, "transactions_v2")
    transactions = data.get("data", {}).get("items", [])
    
    total_volume = 0
    tx_count = len(transactions)

    # Sum the value of all transactions in ETH
    for tx in transactions:
        total_volume += float(tx.get("value", 0)) / 10**18  # Convert wei to ETH
    
    # Risk assessment (adjust the thresholds to Ethereum-specific logic)
    frequency_score = 1 if tx_count < 10 else 3 if tx_count < 50 else 5
    volume_score = 1 if total_volume < 10 else 3 if total_volume < 100 else 5

    return {
        "tx_count": tx_count,
        "total_volume": total_volume,
        "frequency_score": frequency_score,
        "volume_score": volume_score
    }







# Interaction with Known Malicious Addresses
@app.get("/wallet/{wallet_address}/malicious_interactions/")
def malicious_interactions(wallet_address: str):
    # Fetch Ethereum transactions
    data = call_covalent_api(wallet_address, "transactions_v2")
    # print(data)
    # print('*'*100)
    transactions = data.get("data", {}).get("items", [])

    malicious_interaction_count = 0
    malicious_details: List[Dict] = []
    malicious_addresses = {entry["address"]: entry for entry in known_malicious_addresses}

    # Check if any transaction interacts with known malicious addresses
    print(transactions)
    for tx in transactions:
        to_address = tx.get("to_address")
        if to_address in malicious_addresses:
            # Increment malicious interaction count
            malicious_interaction_count += 1
            # Get the associated comment and date from the malicious address list
            malicious_entry = malicious_addresses[to_address]
            malicious_details.append({
                "malicious_address": to_address,
                "comment": malicious_entry["comment"],
                "date_marked_suspicious": malicious_entry["date"]
            })

    # Calculate a malicious score based on interaction count
    malicious_score = 1 if malicious_interaction_count == 0 else 3 if malicious_interaction_count < 5 else 5

    return {
        "malicious_interaction_count": malicious_interaction_count,
        "malicious_score": malicious_score,
        "malicious_details": malicious_details
    }


# Smart Contract Interactions (Ethereum-specific)
@app.get("/wallet/{wallet_address}/contract_interactions/")
def contract_interactions(wallet_address: str):
    # Fetch Ethereum transactions
    data = call_covalent_api(wallet_address, "transactions_v2")
    transactions = data.get("data", {}).get("items", [])
    
    contract_interactions_count = 0
    for tx in transactions:
        if tx.get("to_address_label") == "Smart Contract":
            contract_interactions_count += 1

    contract_score = 1 if contract_interactions_count < 10 else 3 if contract_interactions_count < 50 else 5

    return {
        "contract_interactions_count": contract_interactions_count,
        "contract_score": contract_score
    }




# Cross-chain Activity (For Layer 2 chains like Arbitrum, Optimism)
@app.get("/wallet/{wallet_address}/cross_chain_activity/")
def cross_chain_activity(wallet_address: str):
    # Layer 2 chain IDs: Arbitrum One (42161), Optimism (10)
    l2_chain_ids = [42161, 10]  
    cross_chain_count = 0

    for chain_id in l2_chain_ids:
        try:
            # Call Covalent API for Layer 2 chain
            data = call_covalent_api(wallet_address, "transactions_v2", chain_id=chain_id)
            # Check if there are transactions on this chain
            if len(data.get("data", {}).get("items", [])) > 0:
                cross_chain_count += 1
        except Exception as e:
            # Optionally log the error, but continue the loop
            print(f"Error fetching data for chain_id {chain_id}: {e}")
            continue

    # Score based on cross-chain activity
    cross_chain_score = 1 if cross_chain_count <  4 else 3 if cross_chain_count < 2 else 5

    return {
        "cross_chain_count": cross_chain_count,
        "cross_chain_score": cross_chain_score
    }


# AI powered wallet risk analyser - NOT WORKING 
@app.post("/wallet_risk")
async def fetch_wallet_risk_data(request: WalletRiskRequest) -> dict:
    """Fetch wallet risk data from QuickNode API."""
    address = request.address
    print('API triggered')
    url = f"{QUICKNODE_BASE_URL}{QUICKNODE_API_KEY}/address_label?proto=eth&address={address}"
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        return response.json()  # Assuming the API returns a JSON response
    else:
        return {"error": f"Unable to fetch risk data for address {address}"}


# Helper function to fetch advanced analytics data
async def fetch_advanced_analytics(wallet_address: str):
    # Construct the request payload with JSON-RPC client
    payload = request("sc_getAddressAnalysis", {"hash": wallet_address})
    
    # Make the request to the QuickNode JSON-RPC API
    try:
        response = requests.post("https://dimensional-special-wind.quiknode.pro/ed11912a414200cb1a9bc7b68bed8b69f989f007/", json=payload)
        parsed = parse(response.json())
        
        # Check if the response is successful
        if isinstance(parsed, Ok):
            return parsed.result
        else:
            logging.error(f"Advanced Analytics Error: {parsed.message}")
            return {"error": "Failed to fetch advanced analytics"}
    
    except Exception as e:
        logging.error(f"Exception in Advanced Analytics Request: {e}")
        return {"error": "Exception occurred while fetching advanced analytics"}




# 6. Combined Suspicion Score (Ethereum-focused)
@app.get("/wallet/{wallet_address}/suspicion_scores/")
async def suspicion_score(wallet_address: str):
    # Fetch all the individual scores from their respective endpoints
    tx_activity = transaction_activity(wallet_address)
    malicious_interaction = malicious_interactions(wallet_address)
    contract_activity = contract_interactions(wallet_address)
    cross_chain = cross_chain_activity(wallet_address)
    
    
    # Fetch advanced analytics data asynchronously
    advanced_analytics = await fetch_advanced_analytics(wallet_address)

    # Calculate total score by summing all the individual scores
    total_score = (
        tx_activity["frequency_score"] +
        tx_activity["volume_score"] +
        malicious_interaction["malicious_score"] +
        contract_activity["contract_score"] +
        cross_chain["cross_chain_score"]
    )

    # Normalize the total score to a 0-100 scale
    max_score = 25  # Maximum possible score if all categories are maxed
    normalized_score = (total_score / max_score) * 100

    # Determine risk level based on the normalized score
    risk_level = "High" if normalized_score > 70 else "Medium" if normalized_score > 40 else "Low"

    # Return all the individual scores along with total and normalized score
    return {
        "total_score": total_score,
        "normalized_score": normalized_score,
        "risk_level": risk_level,
        "details": {
            "transaction_activity": {
                "frequency_score": tx_activity["frequency_score"],
                "volume_score": tx_activity["volume_score"]
            },
            "malicious_interactions": {
                "malicious_score": malicious_interaction["malicious_score"],
                "malicious_interaction_count": malicious_interaction["malicious_interaction_count"]
            },
            "contract_interactions": {
                "contract_score": contract_activity["contract_score"]
            },
            "cross_chain_activity": {
                "cross_chain_score": cross_chain["cross_chain_score"],
                "cross_chain_count": cross_chain["cross_chain_count"]
            },
            "advanced_analytics": advanced_analytics  # Include the fetched advanced analytics data
        }
    }

import asyncio
from fastapi.responses import StreamingResponse
connections = []

@app.get("/stream")
async def stream():
    async def event_generator():
        while True:
            if connections:
                message = connections.pop(0)
                message = connections.pop(0)
                save_response_to_file(message)
                yield f"data: {json.dumps(message)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/webhook")
async def quicknode_webhook(request: Request):
    # Receive the data from the request body
    # get list from response body
    

    block = await request.json()
  
    # transactions= block[0].get('content')["transactions"]
    transactions= block[0].get('transactions')
    
    for tx in transactions:
        value_hex = tx['value']
        
        wei_value = int(value_hex,16)  # Convert from hex string to decimal
             
        tx['value_in_eth'] = wei_value/10**18
        

    # Sort transactions by 'value_in_eth' in descending order
    sorted_transactions = sorted(transactions, key=lambda tx: tx['value_in_eth'], reverse=True)
    
    # Find the top 5% transactions
    top_5_percent_count = max(1, len(sorted_transactions) * 5 // 100)  # Ensure at least one transaction is selected
    top_5_percent_transactions = sorted_transactions[:top_5_percent_count]
    # allow only those transaction whose value is greater than 1.5 eth
    top_5_percent_transactions = [tx for tx in top_5_percent_transactions if tx['value_in_eth']>=5]
    # Add to connections for streaming
    result_transactions = [
        {
            "from": tx.get("from"),
            "to": tx.get("to"),
            "value_in_eth": tx["value_in_eth"],
            "hash": tx.get("hash")
        }
        for tx in top_5_percent_transactions
    ]
    connections.append({"result": result_transactions})
        # Prepare data to store in the file
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "transactions": result_transactions
    }

    # Append the response data to `responses.json`
    if os.path.exists("responses.json"):
        with open("responses.json", "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(response_data)

    with open("responses.json", "w") as f:
        json.dump(data, f, indent=4)
    print("========================================================================")
    print(result_transactions)
    # Prepare data to store in the file
    

   

    
    return {"top_5_percent_transactions": result_transactions}


@app.get("/latest_response")
async def get_latest_response():
    # Check if `responses.json` exists
    if not os.path.exists("responses.json"):
        return JSONResponse(content={"error": "No data available"}, status_code=404)
    
    with open("responses.json", "r") as f:
        data = json.load(f)
    
    # Return the last entry in the JSON data
    if data:
        latest_response = data[-1]
        return JSONResponse(content=latest_response)
    else:
        return JSONResponse(content={"error": "No data available"}, status_code=404)







# Z-Score based outlier detection
def z_score_outlier_detection(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    return z_scores

# IQR based outlier detection
def iqr_outlier_detection(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers

# Balance velocity calculation
def balance_velocity(history):
    changes = np.diff(history)
    return changes / np.mean(changes)

# Entropy calculation for randomness of transaction amounts
def entropy_of_transactions(data):
    probabilities = np.histogram(data, bins=10, density=True)[0]
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Calculate Mule Suspicion Score (this can be adjusted based on your scoring logic)
def calculate_mule_score(z_outliers, iqr_outliers, velocity, entropy):
    z_outlier_score = len([z for z in z_outliers if abs(z) > 2.0])  # Number of z-score outliers
    iqr_outlier_score = len(iqr_outliers)  # Number of IQR outliers
    velocity_score = np.mean(velocity) * 10  # Adjust scaling factor for velocity
    entropy_score = max(0, (3 - entropy)) * 10  # The lower the entropy, the higher the score
    
    total_score = z_outlier_score + iqr_outlier_score + velocity_score + entropy_score
    return total_score

# Determine qualitative risk level
def get_qualitative_risk_level(score):
    if score > 70:
        return "High", "This wallet shows highly suspicious activity consistent with mule accounts."
    elif score > 40:
        return "Medium", "This wallet shows moderate suspicious activity and should be further investigated."
    else:
        return "Low", "This wallet shows low levels of suspicious activity."

@app.get("/wallet_balance/{wallet_address}/mule_suspicion_eth/")
async def mule_suspicion(wallet_address: str):
    # Fetch balance history for the past year
    balance_data = await call_quicknode_balance_history(wallet_address)
    
    # Check if the expected fields are in the response
    if 'result' not in balance_data:
        return {"error": "No balance history available for this wallet"}

    # Extract balance history data
    balance_history = []
    for entry in balance_data['result']:
        # Ensure 'sent' and 'received' are valid numbers, and convert them to integers
        sent = int(entry.get('sent', 0))
        received = int(entry.get('received', 0))
        
        # Calculate balance as received - sent
        balance = received - sent
        balance_history.append(balance)

    # If no balance data is present, return an error
    if not balance_history:
        return {"error": "No balance changes found for this wallet"}

    # Calculate balance changes (velocity)
    balance_changes = np.diff(balance_history)

    # Perform statistical analysis on the balance history
    z_outliers = z_score_outlier_detection(balance_changes)
    iqr_outliers = iqr_outlier_detection(balance_changes)
    velocity = balance_velocity(balance_history)
    entropy = entropy_of_transactions(balance_changes)

    # Calculate mule suspicion score
    mule_score = calculate_mule_score(z_outliers, iqr_outliers, velocity, entropy)

    # Normalize the mule score
    normalized_mule_score = (mule_score / 50) * 100

    # Get qualitative risk level and description
    risk_level, risk_description = get_qualitative_risk_level(normalized_mule_score)

    # Prepare detailed scoring explanation
    scoring_explanation = {
        "mule_suspicion_score": {
            "score": normalized_mule_score,
            "interpretation": {
                "0 - 20": "Low suspicion: The wallet shows normal behavior with steady balance changes and no significant outliers.",
                "21 - 50": "Medium suspicion: Some irregularities may be present, such as sporadic high-volume transactions.",
                "51 - 70": "High suspicion: The wallet shows patterns consistent with money mule behavior, such as rapid fund movements.",
                "71 - 100": "Very high suspicion: This wallet exhibits extreme behaviors, such as frequent large transfers that don't correlate with typical transaction patterns."
            }
        },
        "risk_level": risk_level,
        "risk_description": risk_description,
        "details": {
            "z_score_outliers_count": {
                "count": len([z for z in z_outliers if abs(z) > 2.0]),
                "explanation": {
                    "description": "The number of balance changes that fall outside the typical range based on a Z-score threshold of ±2.0.",
                    "normal_range": "0 - 2 outliers is typical for normal wallet behavior, while counts above 5 are considered concerning."
                }
            },
            "iqr_outliers_count": {
                "count": len(iqr_outliers),
                "explanation": {
                    "description": "The number of outliers identified using the Interquartile Range (IQR) method.",
                    "normal_range": "0 - 3 outliers is typical for healthy wallets. Counts above 5 may suggest volatility and require further investigation."
                }
            },
            "velocity": {
                "value": np.mean(velocity),
                "explanation": {
                    "description": "The average velocity of balance changes, calculated as the mean of the differences in balance over time.",
                    "normal_range": "Average velocity under $500 per month is typically considered normal. Above $1000 may indicate suspicious activity."
                }
            },
            "entropy": {
                "value": entropy,
                "explanation": {
                    "description": "A measure of the unpredictability or randomness of the transaction amounts.",
                    "normal_range": "Entropy values between 0 - 0.5 indicate predictable transactions. Values above 1 suggest high randomness, which could be suspicious."
                }
            }
        }
    }

    return {
        "wallet_address": wallet_address,
        **scoring_explanation
    }

# Endpoint to get transaction analysis
@app.get("/transaction/{transaction_hash}/analysis")
async def transaction_analysis(transaction_hash: str):
    # Set up the JSON-RPC request payload
    payload = request("sc_getTransactionAnalysis", {"hash": transaction_hash})
    
    # Make the request to the QuickNode JSON-RPC API
    try:
        response = requests.post("https://dimensional-special-wind.quiknode.pro/ed11912a414200cb1a9bc7b68bed8b69f989f007/", json=payload)
        parsed = parse(response.json())
        
        # Check if the response is successful
        if isinstance(parsed, Ok):
            return parsed.result  # Return the parsed result from QuickNode
        else:
            logging.error(f"Transaction Analysis Error: {parsed.message}")
            raise HTTPException(status_code=500, detail="Error fetching transaction analysis")
    
    except Exception as e:
        logging.error(f"Exception in Transaction Analysis Request: {e}")
        raise HTTPException(status_code=500, detail="Exception occurred while fetching transaction analysis")




# import uvicorn
# if __name__ == "__main__":
    
#     uvicorn.run(app, host='127.0.0.1', port=8000)
