def maxProfit(prices):
      
   minimum=min(prices)
   min_index=prices.index(minimum)
   maxi=min_index
   if min_index == len(prices) - 1:
       return 0
   for i in range(min_index, len(prices)):
       if prices[i] > prices[maxi]:
        
            maxi = i
        
   return prices[maxi] - minimum    
    
       
l1=[7,1,5,3,6,4]
print(maxProfit(l1))    