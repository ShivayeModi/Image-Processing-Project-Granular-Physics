size(a) //display rows and columns
[max_a, imax] = max(abs(a(:))) //max_a means max value
//imax means element number in the matrix(integer)
//formula for imax is given by
//imax=(no.of rows)*(column no.-1)+row.no
//note:the first element has location (1,1)
//ypeak,xpeak provides location of maxima
//size(a,1) provide rows
//size(a,2) provide columns