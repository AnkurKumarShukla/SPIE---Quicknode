import requests
import json

url = "https://dimensional-special-wind.quiknode.pro/ed119******************/"

payload = json.dumps({
  "method": "bb_getBalanceHistory",
  "params": [
    "0x9862f**************e53d792ded66eb",
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

response = requests.request("POST", url, headers=headers, data=payload)

print(response.json())

# 0x9862fd5b8db********************ded66eb
# balance wala address

# from jsonrpcclient import request, parse, Ok
# import logging
# import requests
# response = requests.post("https://dimensional-special-wind.quiknode.pro/ed1191***************************/", json=request("sc_getAddressAnalysis", { "hash": "0xa7efae728d2936e78bda97dc267687568dd593f3" }))
# parsed = parse(response.json())
# if isinstance(parsed, Ok):
#     print(parsed.result)
# else:
#     logging.error(parsed.message)
