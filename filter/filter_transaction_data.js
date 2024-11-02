function main(data) {
    var filteredData = data.streamData;
  
    // Create a new array for transactions where value is greater than 0x0
    var filteredTransactions = filteredData.transactions.filter(function(transaction) {
      return transaction.value !== "0x0";
    });
  
    // Return the entire object but with filtered transactions
    return {
      ...filteredData,  // Spread the existing properties of filteredData
      transactions: filteredTransactions  // Override the transactions with the filtered ones
    };
  }