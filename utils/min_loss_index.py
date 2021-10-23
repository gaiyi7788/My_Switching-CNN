def min_loss_index(loss1,loss2,loss3):
    loss = [loss1.item(),loss2.item(),loss3.item()]
    min_loss = min(loss)
    if min_loss == loss1.item():
        return loss1, 1
    elif min_loss == loss2.item():
        return loss2, 2
    elif min_loss == loss3.item():
        return loss3, 3
    else:
        return 0