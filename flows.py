
import torch.nn as nn

import torch.nn.functional as F

import torch



class Tnn1(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder1 = nn.Linear(20,20, bias=False)


	def forward(self, src):
		output = self.encoder1(src)

		output = torch.tanh(output)

		return output




class Ende(nn.Module):
	def __init__(self):
		super().__init__()
		self.tnn1 = Tnn1()
		self.tnn2 = Tnn1()
		self.tnn3 = Tnn1()
		self.tnn4 = Tnn1()


	def forward(self,src_inputt):#batch 1 40
		a = src_inputt.shape[2] # 40
		b = int(a / 2) #20

		p1 = src_inputt[:, :, 0:b] #p1 batch 1 20
		p2 = src_inputt[:, :, b:]  #p2 batch 1 20

		f1_p1 = self.tnn1(p1) # batch 1 20


		f2_p1 = self.tnn2(p1)
		exp_f2_p1 = torch.exp(f2_p1)
		p2_exp = torch.multiply(p2,exp_f2_p1)#batch 20 1

		s1 = p1   # batch  1 20
		s2 = p2_exp+f1_p1

		for i in self.tnn1.parameters():
			w1 = i

		for i in self.tnn2.parameters():
			w2 = i

		I = torch.eye(b,b).repeat(src_inputt.shape[0],1,1).to(src_inputt.device)
		zeros = torch.zeros(b,b).repeat(src_inputt.shape[0],1,1).to(src_inputt.device)
		one = torch.ones(1,b).repeat(src_inputt.shape[0],1,1).to(src_inputt.device)
		w11 = w1.repeat(src_inputt.shape[0],1,1).to(src_inputt.device)
		w22 = w2.repeat(src_inputt.shape[0], 1, 1).to(src_inputt.device)


		J22 = torch.multiply(torch.bmm(one.transpose(1,2),exp_f2_p1),I)

		jls1_11 = torch.bmm(one.transpose(1,2),one - torch.multiply(f1_p1,f1_p1)).transpose(1,2)
		J21_11 = torch.multiply(jls1_11,w11)



		J21_1 =  J21_11


		Jls = torch.bmm(one.transpose(1,2),p2_exp).transpose(1,2)
		jls1_2 = torch.bmm(one.transpose(1,2),one - torch.multiply(f2_p1,f2_p1)).transpose(1,2)
		J21_2 = torch.multiply(Jls,torch.multiply(jls1_2,w22))

		J211 = J21_2+J21_1

		J21 = J211
		J1_1 = torch.cat((I, zeros),2)
		J1_2 = torch.cat((J21, J22), 2)
		J_1 = torch.cat((J1_1, J1_2), 1)
#-=======================================================================================================

		f3_s2 = self.tnn3(s2)  # batch 20 1
		f4_s2 = self.tnn4(s2)
		exp_f4_s2 = torch.exp(f4_s2)
		s1_exp = torch.multiply(s1, exp_f4_s2)  # batch 20 1

		t1 = s1_exp + f3_s2  # s1_exp
		t2 = s2

		src_inputt11 = torch.cat((t1, t2), 2)


		for i in self.tnn3.parameters():
			w3 = i

		for i in self.tnn4.parameters():
			w4 = i



		w33 = w3.repeat(src_inputt.shape[0],1,1).to(src_inputt.device)
		w44 = w4.repeat(src_inputt.shape[0], 1, 1).to(src_inputt.device)


		JJ11 = torch.multiply(torch.bmm(one.transpose(1,2),exp_f4_s2),I)

		jls2_21 = torch.bmm(one.transpose(1, 2), one - torch.multiply(f3_s2, f3_s2)).transpose(1, 2)
		JJ12_21 = torch.multiply(jls2_21,w33)

		JJ12_2 = JJ12_21


		Jls2 = torch.bmm(one.transpose(1, 2), s1_exp).transpose(1, 2)
		jls2_1 = torch.bmm(one.transpose(1, 2), one - torch.multiply(f4_s2, f4_s2)).transpose(1, 2)
		JJ12_1 = torch.multiply(Jls2,torch.multiply(jls2_1, w44))

		JJ12 = JJ12_1+JJ12_2#
		JJ1_1 = torch.cat((JJ11, JJ12), 2)
		JJ1_2 = torch.cat((zeros, I), 2)
		J_2 = torch.cat((JJ1_1, JJ1_2), 1)
#================================================================================



		J_T = torch.bmm(J_2,J_1)

		return src_inputt11,J_T



class TransAm(nn.Module):
	def __init__(self,feature_size=1, num_layers=1, dropout=0.):
		super(TransAm, self).__init__()
		self.bias = None
		self.encoder1 = nn.Linear(2, 40, bias=False)
		self.linear2 = nn.Linear(1, 1, bias=False)

		self.ende = Ende()


	def forward(self, src):
		src_input = src  # 1 batch 2
		src_inputt = self.encoder1(src_input) #1 batch 40
		#
		wg_sf = self.encoder1.weight # 40 2

		src_inputt = src_inputt.transpose(0,1)
		src_inputt11,J_T = self.ende(src_inputt)

		src_inputt11 = src_inputt11.transpose(0,1)
		src_inputt111 = self.linear2(src_inputt11.transpose(0,2)).transpose(0,2)
		www2 = self.linear2.weight
		# for i in self.linear2.parameters():
		# 	i.data = torch.clamp(i,-0.001,0.001)


		ls2 = www2
		lss2 = F.relu(-ls2) / ls2
		output4_linear = src_inputt111 + 2 * lss2 * src_inputt111
		output4 = output4_linear#

		output55 = -output4 # 1 batch 40


		wgg = torch.mm(wg_sf.T, wg_sf) #2 2
		wgg_inv = torch.inverse(wgg) # 2 2
		wg_sf_inv = torch.mm(wg_sf,wgg_inv) # 40 2

		# hat = torch.mm(wg_sf_inv,wg_sf.T)

		output555 = torch.bmm(output55.transpose(0,1), J_T)

		output555_2 = torch.mm(output555.squeeze(1), wg_sf_inv).unsqueeze(1)


		output5 = output555_2.transpose(0,1)

		look = output5

		# look_11 = torch.mm(look.squeeze(0),wg_sf.T)
		#
		# look_22 =torch.bmm(torch.bmm(look_11.unsqueeze(1),J_T.transpose(1,2))
		# 				   ,src_inputt11.transpose(0,1).transpose(1,2)).squeeze(1).squeeze(1)
		#
		# for i in look_22:
		# 	if i >0:
		# 		print("wrong")



		return look,wg_sf.T,output555.transpose(0,1)




