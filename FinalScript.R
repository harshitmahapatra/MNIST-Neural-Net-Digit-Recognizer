#Reading the data and splitting it into training and cross #validation set
train <- read.csv("mnist_train.csv",header= FALSE)
splittingseed <- 123
set.seed(splittingseed )
selsize <- floor(0.70*nrow(train))
sel_ind <- sample(seq_len(nrow(train)),size=selsize)
trainset <- train[sel_ind,]
cvset <- train[-sel_ind,]
X <- trainset[,-1]
Y <- trainset[,1]
trainlabel <- trainset[,1]
cvlabel <- cvset[,1]
library(nnet)
Y <- class.ind(Y)
print(X[1:5,1:5])
print(Y[1:5,])

#Functions to help reduce the dimensions of images
reducematrix <- function(source_matrix, red_ratio) {
  if(red_ratio == 1) {
    return(source_matrix)
  }
  red_ratio <- 1/red_ratio #reduction ratio
  red_matrix <- matrix(0, nrow = nrow(source_matrix)/red_ratio, ncol = ncol(source_matrix)/red_ratio) #new matrix with reduced dimensions
  for(i in 1:nrow(red_matrix))
  {
    for(j in 1:ncol(red_matrix))
    {
      red_matrixrow <- c((red_ratio*i - red_ratio + 1):(red_ratio*i)) #figuring out which pixels get merged
      red_matrixcolumn <- c((red_ratio*j - red_ratio + 1):(red_ratio*j))
      for(x in red_matrixrow){
        for(y in red_matrixcolumn){
          red_matrix[i,j] <- red_matrix[i,j] + source_matrix[x,y]/(red_ratio*red_ratio) #merging the pixels
        }
      }}}
  return(red_matrix)
}
reduceDataFrame <- function(frameJob, red_ratio, hasLabels) {
  adjust <- 0
  if(hasLabels == TRUE) {
    adjust <- 1
  }
  dimension <- sqrt(ncol(frameJob) - adjust) #assume square matrix
  newColSize = dimension*dimension*red_ratio*red_ratio + adjust #new column size with reduced dimensions
  newFrame <- data.frame(matrix(0, ncol = newColSize, nrow = nrow(frameJob))) #new frame with reduced dimensions
  for(z in 1:nrow(frameJob)) {
    if((z %% 1000) == 0) {
        cat(c(z, "\n"))
      }
imatrix <- matrix(0, nrow = 2, ncol = 2) #matrix to hold image
#test vs train
if(adjust == 0) {
  imatrix <- matrix(frameJob[z, ], ncol = dimension)
}
if(adjust == 1) {
  imatrix <- matrix(frameJob[z, -1], ncol = dimension)
}
imatrix <- as.character(imatrix)
imatrix <- as.numeric(imatrix) 
imatrix <- matrix(imatrix, ncol = dimension)
imatrix <- reducematrix(imatrix, red_ratio)
newRow <- c(as.vector(imatrix))
if(adjust == 1) {  #add labels back in
  newRow <- c(frameJob[z, 1], newRow)
}
newFrame[z, ] <- newRow
  }
  return(newFrame)
}
#End of functions

#Applying the function to training set and also normalizing it
X_reduced <-reduceDataFrame(X,red_ratio=1/2,hasLabels=FALSE)
X_reducednormalized <- X_reduced/255
print(X_reducednormalized[1:5,1:5])


#Visualizing original and reduced image
attach(X)
par(mfrow=c(2,2),mai=c(0.1,0.1,0.1,0.1))
m <- matrix(unlist(X[3,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)
m <- matrix(unlist(X[12,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)
m <- matrix(unlist(X[48,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)
m <- matrix(unlist(X[212,]),nrow = 28,ncol = 28 )
image(m,axes=FALSE)
par(mfrow=c(1,2),mai=c(0.8,0.5,0.5,0.5))
m<- matrix(unlist(X[3,]),nrow = 28,ncol = 28 )
image(m,xlab = '(a)')
m <- matrix(unlist(X_reduced[3,]),nrow = 14,ncol = 14 )
image(m,xlab = '(b)')


#Applying the function to cross validation set and normalizing it
cv_reduced <-reduceDataFrame(cvset[,-1],red_ratio=1/2,hasLabels=FALSE)
cv_reducednormalized = cv_reduced/255
print(cv_reducednormalized[5,5])

#Forming a data table to store error for various combinations of #no of nodes and maximum iterations.
datatable_no_of_nodes <- data.frame(c(10,15,20,25,30,40,50),c(100,100,100,100,100,100,100),as.numeric(c(NA,NA,NA,NA,NA,NA,NA)),as.numeric(c(NA,NA,NA,NA,NA,NA,NA)),as.numeric(c(NA,NA,NA,NA,NA,NA,NA)))
colnames(datatable_no_of_nodes) <- c("no_of_nodes","maxiter","Accuracytrain","AccuracyCV","Seed")
print(datatable_no_of_nodes)

#Function to train a neural network (where the no_of_nodes and maxiter are taken from a row of datatable) and the calculated #accuracies are stored in the same row.

calculate_accuracy<- function(i,nodes,iter)
{
  require(nnet)
  s <-123+i
  set.seed(s)
  cat(sprintf("Nodes:%f\n",nodes))
  cat(sprintf("Maxiter:%f\n",iter))
  model <- nnet(X_reducednormalized,Y,size=nodes,softmax=TRUE,maxit=iter,MaxNWts =80000)
  prediction <- predict(model,X_reducednormalized,type="class")
  table(prediction,trainlabel)
  correct <- prediction==trainlabel
  AccuracyTrain <- (sum(correct)/nrow(X_reducednormalized))*100
  cat(sprintf("Accuracytrain:%f\n",AccuracyTrain))
  prediction2 <- predict(model,cv_reducednormalized,type="class")
  table(prediction2,cvlabel)
  correct2<- prediction2==cvlabel
  AccuracyCV <- (sum(correct2)/nrow(cv_reducednormalized))*100
  cat(sprintf("Accuracycv:%f\n",AccuracyCV))
  return(c(AccuracyTrain,AccuracyCV,s))
}
#End of function

#Applying the function to all the rows of datatable 
for(j in 1:7)
{
  temp <-calculate_accuracy(j,datatable_no_of_nodes[j,1],datatable_no_of_nodes[j,2])
  datatable_no_of_nodes[j,3]<-temp[1]
  datatable_no_of_nodes[j,4]<-temp[2]
  datatable_no_of_nodes[j,5]<-temp[3]
}
#Adding columns for errors in data table
datatable_no_of_nodes$ErrorTrain <- (100-datatable_no_of_nodes$Accuracytrain)
datatable_no_of_nodes$ErrorCV <- (100-datatable_no_of_nodes$AccuracyCV)
print(datatable_no_of_nodes)


#Storing the datatable in png format for future reference.
library(gridExtra)
png("datatable.png",height = 300,width = 4000)
p <-tableGrob(datatable_no_of_nodes)
grid.arrange(p)
dev.off()




#Ploting the graph of no_of_nodes vs errors
plotdataframe <- datatable_no_of_nodes[,c(1,6,7)]

#melting the dataframe to feed to ggplot() function
library(reshape2)
meltedframe <- melt(plotdataframe,id="no_of_nodes")

#Applying ggplot() function
library(ggplot2)
finalplot<-ggplot(meltedframe,aes(no_of_nodes,value,color=variable))+geom_line()+scale_colour_manual(values = c("red","blue"))
finalplot <- finalplot+xlab("No of nodes")+ylab("Error Value")+ggtitle("Plot of No of Nodes vs Error(Maxiter=100)")
finalplot

#From the plot we choose no_of_nodes=50 and maxiter=100 for #optimal error

#Train nnet using optimal parameters on trainingset+cvset to form #final training set
totaltrain_reducednormalized <- rbind(X_reducednormalized,cv_reducednormalized)
totaltrainlabel <- c(trainlabel,cvlabel)
totaltrainfinal <- class.ind(totaltrainlabel)

#Training the nnet on totat_training_set
finalseed <- 150
set.seed(finalseed)
model_final <- nnet(totaltrain_reducednormalized,totaltrainfinal,size=50,softmax=TRUE,maxit=100,MaxNWts = 80000)


#Reading,reducing and normalizing test data
test<- read.csv("mnist_test.csv",header=FALSE)
test_reduced <-reduceDataFrame(test,red_ratio=1/2,hasLabels=TRUE)
test_reducednormalized <-test_reduced[,-1]/255

#Calculating Final Accuracies
prediction2 <- predict(model_final,test_reducednormalized,type="class")
correct <- prediction2==test_reduced[,1]
FinalAccurracyTest <- sum(correct)/nrow(test_reduced) *100
prediction <- predict(model_final,totaltrain_reducednormalized,type="class")
correct <- prediction==totaltrainlabel
FinalAccurracyTrain <- sum(correct)/nrow(totaltrain_reducednormalized) *100
cat(sprintf("Final train accuracy =%f\n",FinalAccuracyTrain))
cat(sprintf("Final test accuracy =%f\n",FinalAccuracyTest))

#Final TestAccuracy=96.08
#Final TrainAccuracy=99.27


