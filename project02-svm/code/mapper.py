#!/usr/bin/env python2.7

import sys

import numpy as np


means = np.fromstring("0.0021246993507595866 0.0032713784391997795 0.0033522806931598247 0.0017432450192796278 0.0034743692038798537 0.003637888546929857 0.0019210039127597624 0.0021841610994196136 0.0018762718393396005 0.0034590054363498003 0.0052604099446999682 0.004508790286140099 0.0035272400244497799 0.0030404807453598324 0.0022447918038096385 0.0017851536926196112 0.0021643550482296344 0.0037976255097098874 0.0025753731081197833 0.0029230906247597055 0.0060828219621099217 0.0023575999971396813 0.0043864294801700945 0.0071589655821691772 0.0036986840015399082 0.00057556662468004468 0.0030184163825898096 0.0062797556933995476 0.0018388575003994976 0.0018222650139394971 0.0032805952842698042 0.0035132540814598752 0.0024659598304896477 0.0026319448493497136 0.003572205969799843 0.0030648003435798008 0.0021365654833496528 0.0012356635529695108 0.0021261889005796605 0.0030134591283298012 0.0016100815367798148 0.012523000339860027 0.002519218599329652 0.0052571679389798714 0.0026606913287896975 0.0028296754183797139 0.0039323969569099605 0.0020691205227195992 0.0030826525382697508 0.0020232189983895653 0.0040679867872599708 0.0018371556472196301 0.0031808009477497599 0.0034889724135098699 0.0041241983089198644 0.003466312111199805 0.00070525738208999413 0.0012962120699994075 0.0023748498468496439 0.0039468429845199238 0.0024428431670496745 0.012215355168679928 0.0012535008249493743 0.0026764566235297597 0.0043243784063398552 0.00065200872076008631 0.0022265717804095869 0.00081018893256987797 0.0027757838127496974 0.0011937874021293784 0.0033124457059298595 0.0033779817461398022 0.0026583629339898352 0.00096654598538961438 0.0021773139189896237 0.002624655562289701 0.0015705430665195477 0.0030252402714297136 0.0040940954038199478 0.0027594978981697318 0.00079096095234988185 0.0026036506797997572 0.0027190828795197546 0.0027920414767097406 0.0018699793252895133 0.002401434445989645 0.0031948320317497989 0.0028928477797297309 0.001254727068959367 0.0022096979193596154 0.0021709718136396741 0.0022725767293796106 0.0036734258169697923 0.0028088068982497589 0.00058128786511008252 0.0030860261422598389 0.0028005311404197221 0.0013144850578592786 0.00075680244248994735 0.003594669478579891 0.0032807255223097792 0.0023280524667396774 0.00318162350717981 0.0038591178877899067 0.0027019215482496691 0.00097254474824969451 0.0023080437106096615 0.0013613457456093684 0.0045951612643399054 0.0038485342457099387 0.00043219164003003777 0.001528150938759669 0.0024822021413396867 0.0018061700621995042 0.0028432498431096936 0.00055539853847006056 0.004173783897349969 0.0023134058954397316 0.0035923805665898493 0.001944158411359583 0.0022174885522996423 0.0021200232347196586 0.0014086675440495285 0.0060588732600395838 0.0020999206563196006 0.002311535350179601 0.0012081675861494046 0.0029662122591298679 0.0023064668532896651 0.00086526146860972403 0.0035453290259598483 0.0022721631862096265 0.003677016888759915 0.0027193153269396897 0.0019698620481495626 0.002072663196939612 0.0014700221401894075 0.00017158202360999703 0.0022463464680696336 0.0035194326419099174 0.0030686680423197867 0.002374867405639663 0.00042710055163003362 0.0030035550561797468 0.0037270432987298683 0.0024282900953096712 0.00086048954793975898 0.0011186456857096038 0.0027912283038996942 0.0013746583237494142 0.0043072999357398533 0.0032034503423598666 0.0025760441755196838 0.0023421858856196836 0.002131599313139612 0.0029099423010796777 0.0016998768135196812 0.0028229397603697181 0.0030535556897598208 0.003180828002529861 0.0041489816552998261 0.00056885910910004086 0.0030288286590998306 0.0002859839918500021 0.0025907458249397565 0.0019840401991995621 0.0026709580203396733 0.00018365706286999837 0.0023102737736697076 0.0019214511389595858 0.0035872736249698512 0.0030397738456597189 0.00058895044087008347 0.0011302665188195724 0.0036135427626998772 0.00096930456685965713 0.0018706273234795688 0.0028471338214996859 0.0040263350593498478 0.00041504772780003257 0.00088363138039978097 0.0027967429290597077 0.0021579785680196756 0.0032100556617598404 0.0012821952431594156 0.0011697489935395071 0.0024514963691797428 0.0024098468797296444 0.0035879574826698079 0.003169685177989759 0.0053416716965498916 0.0031958328667698248 0.003017081933489743 0.00025151594039000199 0.0022886760678696417 0.0022956408480896266 0.0041254424031998971 0.0023694221563096735 0.0047916681473398276 0.00051616048678002784 0.0009364954557196728 0.00036740167022002141 0.00078959433233993142 0.0007410161818699483 0.0028233597298397656 0.0035765694441198263 0.0057271246152496317 0.0031925037529198339 0.0014168537242193022 0.0040282638127298667 0.0027408330144697043 0.0022817147531596685 0.0013110019340695283 0.0031049698000498423 0.0014794847673093696 0.0019060075812395761 0.0023860511557697102 0.0025873434738996485 0.0018797211826496064 0.0022561836261797042 0.0041991871207300085 0.0023698767044296855 0.0022702994190196093 0.0029535219055797368 0.00079702808800984168 0.00017141315798999718 0.00071072931258999632 0.0022027503444296218 0.0023522028982396696 0.00034261818457001714 0.0030124186968896794 0.0040563730303498731 0.0018014168708095377 0.0045389503904098493 0.0025631905209596659 0.0021709940360196437 0.0031014667275497628 0.0013724805472092871 0.0011206960384995625 0.0035493743115597959 0.0025190975770797062 0.0042803605014598489 0.0009058812431398496 0.0015261301214595528 0.00043206103726003953 0.002057161621769605 0.0029676093005998037 0.00059245340563008307 0.00060303803797007931 0.0055309290333298089 0.0006063130560400961 0.0024001375326397033 0.0051570050648799921 0.003216390780179791 0.001684353076369542 0.0024379539857596923 0.0033096221900098537 0.001808786421229587 0.00036056760674001951 0.0022999638755596282 0.0042300521607298008 0.0033374014801298532 0.0041061077925497727 0.00090300806356967953 0.0027771389140698217 0.0020966763969595594 0.0022364965134396191 0.0021630341014396426 0.003362866027789783 0.0025631540862897312 0.004191926116449857 0.0023811464991296992 0.0013004726735392649 0.0038548863857898333 0.0021571933421396868 0.0032544925816697214 0.0024967996225797357 0.0039128733433798774 0.0032033546653597454 0.0035349716580698469 0.0022774309789496266 0.0020827816616296431 0.001280163236199224 0.0029566993924298487 0.0030278382394197082 0.0031458574724698 0.00073484413224997748 0.0023053594018396508 0.0032629178035998552 0.0028317322999097433 0.0033847674035998084 0.0021507182045496622 0.0013635142890994728 0.0025417732184397166 0.00046798538031004748 0.0014196850140693168 0.001775496716359453 0.00041823802366003517 0.00072539019745996431 0.0030393665008997704 0.0013936213581092793 0.0024650105378997201 0.0002448311107500009 0.0043274930097698871 0.0045290280761799487 0.0047295668273101684 0.0010168427077595955 0.0027754963934396339 0.0028934546900597821 0.0024947583902996968 0.0017947966152195337 0.002808371739829744 0.00080562592018981933 0.0014184058297892733 0.0018558152750695453 0.0018534208896895739 0.0027403346575797425 0.0031581041628497997 0.0019250669095596151 0.0017553527272695774 0.002912743471719791 0.00051881062016005577 0.0041509390442198381 0.0013269250644194269 0.002515913493569724 0.0032034703723998357 0.0015867479873494805 0.0033147417203898185 0.0032343107633697474 0.0016084849715195411 0.00041333437351003248 0.0015982072633194113 0.0014028860576195891 0.0022158183125796393 0.0029487353931697447 0.0028615529172198303 0.0012540566466694289 0.0028261495420197243 0.00017822631116999813 0.0014531231202394163 0.0025906615127396855 0.0036318312786498171 0.002825987395589701 0.0032132990932597881 0.00093148496318973544 0.0022986618991797251 0.0031201742482197584 0.0088757592945090114 0.0019739854059195429 0.0015964743898695729 0.0030620168350797899 0.0017549143672195243 0.0025403744949397296 0.0013998610671793503 0.00050658872377004334 0.0024219329259397276 0.0016578000335194041 0.0014255931402395057 0.0025947821308797258 0.0015455710208097471 0.0019424337106196282 0.0043638276133198444 0.0024791513534598046 0.002471546965979776 0.0032594199180097532 0.00081732890395981583 0.00047798563291005168 0.0026265644132597047 0.0029957660721997665 0.0033466747844698567 0.0030434931783497998 0.0032186603864098446 0.0025580746428896777 0.0074381240438289309 0.0026177068932397522 0.0010374525766094667 0.0018484145568895259 0.0032105816832397539 0.0025588880273796702 0.0011027058149395553 0.00165028316301944 0.0022621210840096185 0.00063843135713010388 0.002677249425599694 0.0011529594838495104 0.0020757956716295806 0.00063164132836008679 0.0012984328854694727 0.0030668599805997697 0.0013209850432293402 0.0017350537225995246 0.0027999960618096992 0.0045968238896799086 0.0015396509469794125 0.0026842448170297231 0.0020969214423495791 0.0032249556936598013 0.00029111348006000424 0.0011551860431694666 0.0031812251568797824", sep=" ")
vars = np.fromstring("3.8753948237858108e-06 1.2972946111794674e-05 1.2594051521366083e-05 5.0841523278404734e-06 1.8774317409263048e-05 6.2913210996917487e-05 1.269807222669888e-05 3.2193349475262057e-06 6.5226200570272061e-06 1.1473588836338628e-05 2.7180466935587737e-05 1.4762302565458717e-05 3.3722317512532468e-05 6.8216505240041436e-06 7.1028116499628903e-06 6.5493827073439618e-06 3.80367131264172e-06 1.4028847130371071e-05 9.3773632055309283e-06 6.493323349342037e-06 0.0012533506935897218 4.9911335763841195e-06 1.2793399333055094e-05 7.251611930188133e-05 9.5489822043414659e-06 3.8895300628186868e-06 4.173457402556971e-05 0.00011347419063456421 2.5715278760111459e-06 3.2518257183024889e-06 1.1746203655396577e-05 5.564016383146592e-05 3.6296631509353909e-06 4.3289811407316681e-06 1.6025500646546836e-05 8.7246747361516438e-06 4.2410327327645271e-06 4.3732089713098806e-06 5.9073865563619062e-06 2.4944097977347468e-05 2.6986158170267078e-05 0.00019357426874984057 5.1764074423215301e-06 3.5213588425492417e-05 5.6548098935816624e-06 4.9935937088475483e-06 2.3828362907972465e-05 3.521023866293484e-06 4.9870702736337188e-06 2.7658266039366798e-06 1.1424139609302174e-05 4.6380793952958809e-06 8.1857174384998292e-06 1.6642225648910047e-05 1.8268643132929127e-05 1.5473118685259949e-05 9.7616078787441458e-07 4.1097607144367696e-06 5.0459663323074957e-06 8.1752036387080678e-06 6.2517426726346483e-06 0.00021128251533625498 2.4441154311918049e-06 1.0193291769369655e-05 1.6000078417860217e-05 1.3360615760691735e-06 3.9318274983244583e-06 3.7424801978201094e-06 4.5859948912655592e-06 2.1863893895928264e-06 1.4465960374765088e-05 1.226800721873276e-05 1.8464105024954982e-05 1.6648636202068534e-06 3.6936226607579947e-06 6.5624020308052344e-06 8.1339303452353934e-06 9.5047711128428641e-06 1.4246167594118415e-05 6.0140973294197884e-06 1.8256200156735017e-06 1.0903757639504039e-05 5.5080914174679564e-06 5.2142169736994904e-06 2.6292604236996645e-06 4.9623024158512934e-06 1.3171420269231491e-05 5.1064782563443342e-06 2.2201233797532346e-06 3.5523146873797785e-06 4.0447453033151591e-06 3.4393314844283629e-06 1.2283374778942664e-05 1.2292876875817127e-05 1.3500473667799135e-06 1.5982740863426082e-05 5.1149263226338105e-06 3.6545265412690049e-06 4.4324293930103502e-06 1.3464151551507424e-05 8.2607323905827565e-06 5.3487969307959027e-06 7.699747933440781e-06 2.6028092053793074e-05 4.6160336251911396e-06 3.4679078250202434e-06 4.1733322591036512e-06 5.3685295356327671e-06 4.1690461279070458e-05 1.8175584863744415e-05 1.4529974714941822e-06 8.9646541680474962e-06 3.8638936584656166e-06 2.9622882868516527e-06 7.0496709821419259e-06 3.1582263769680431e-06 9.405912339046591e-06 9.0755581225100531e-06 1.6325319116706371e-05 7.4249528783198223e-06 6.4142049677004635e-06 4.5308256388559377e-06 4.3379101302365048e-06 0.00010082767573262403 3.8073220474859233e-06 3.2462395975613701e-06 2.7311928376618711e-06 1.3798802536934602e-05 4.3141812822167945e-06 1.3418830948478911e-06 1.2429912124862659e-05 4.5075176921294976e-06 5.1646366657811792e-05 4.5044907401523191e-06 3.8984503442526084e-06 3.5443432542494581e-06 2.4525978397502771e-06 9.3143290305042167e-07 5.5977615024444758e-06 1.4190797086073543e-05 6.9561233764939789e-06 8.0114861452901582e-06 1.2454920191746878e-06 5.5587154982870272e-06 1.0799672251505274e-05 5.2959102834492533e-06 1.5685688647449261e-06 4.0529428722210623e-06 1.1678512895855624e-05 3.2192802988981066e-06 3.7209970472627806e-05 1.3342539819491425e-05 7.8622903069455567e-06 5.2192321914900928e-06 3.9052134579505441e-06 5.0680769571043553e-06 7.9552828837898563e-06 6.7762118492538826e-06 8.5875102642240075e-06 2.0992545616427373e-05 2.0487505271743291e-05 4.3745997535029968e-06 7.1046977878669946e-06 7.7167495498190023e-07 1.0141932308464567e-05 4.2219873766408028e-06 4.5710190852658248e-06 1.1970402654479661e-06 7.6102614732724262e-06 4.6239298630603015e-06 4.9995946799371758e-05 5.7956634809724437e-06 1.5024720589152287e-06 6.0635032731039673e-06 1.5391627780641011e-05 2.178652052162647e-06 4.2030056647134055e-06 5.0822379579415565e-06 1.9836303495641017e-05 1.8930994307717652e-06 3.0604158961858623e-06 6.5280625603827021e-06 1.0265727137904331e-05 4.0302422231094213e-05 3.7750836192671517e-06 4.0367914908354297e-06 8.0446362665366717e-06 4.7656248380853414e-06 8.6978972436276061e-06 7.9679700766206762e-06 2.9451374286812033e-05 1.3111273739035649e-05 6.6028118897700181e-06 1.4941804584231896e-06 3.9528326512917906e-06 3.847295383196301e-06 3.5756600152130488e-05 6.10565382283349e-06 3.5891435340776665e-05 3.6066217532076844e-06 2.0888559126779404e-06 1.0755002920858641e-06 5.8998610038911923e-06 1.9512692088167549e-06 5.0713400804749472e-06 1.4585512351101608e-05 4.822908984311966e-05 9.8016096252778945e-06 1.9911328814957375e-06 2.4764204976600043e-05 6.4805250037636707e-06 3.7935478658080509e-06 5.1083549212952252e-06 1.1189053457458745e-05 2.5200508287861594e-06 7.8373349366817099e-06 7.9847294470099685e-06 4.3095275213819756e-06 5.0268163315597379e-06 7.4832981742681862e-06 9.2408501776852945e-06 4.784135850487231e-06 5.6532252724841891e-06 5.3930817570733614e-06 1.2687973462442569e-06 1.0372124095824449e-06 4.1435096417718113e-06 3.9981959056867675e-06 4.3520178967986713e-06 1.4659748060826231e-06 5.3366902864809163e-06 3.1416765924193689e-05 2.9749844077512922e-06 3.1381515491784522e-05 4.3260417959669591e-06 4.197030498717592e-06 5.306570430382929e-06 1.8883854746421685e-06 4.1937519548496871e-06 6.9194038197555032e-06 7.1767073252994241e-06 2.4833484439498967e-05 5.8383610252210572e-06 3.6243330253608428e-06 1.18902799300137e-06 3.8963636200265115e-06 1.2883918919165478e-05 1.3605525456692033e-06 1.7407965336936251e-06 0.00016857768522088627 1.4100686994311071e-06 7.1668903489840609e-06 2.7108318215380169e-05 5.4590558436375845e-06 6.2033647867466643e-06 5.6859033955868132e-06 4.3241078188076546e-05 9.0432098151017242e-06 1.0594888618529579e-06 4.0484870451845699e-06 2.9548153849811755e-05 9.410471996079331e-06 2.1009809505791367e-05 1.4939978216919125e-06 2.1026313371938338e-05 2.912760631269843e-06 4.1130865661336849e-06 4.0964425120045752e-06 1.0334704132812778e-05 1.1639088987295558e-05 2.0866544215744135e-05 4.7665503013673322e-06 2.4282885077105844e-06 2.4696946110127049e-05 5.8943453758772547e-06 7.0559765519393299e-06 8.6495232917104309e-06 2.4674527585132413e-05 6.5466440985476235e-06 1.4291488938382783e-05 4.0363838996778781e-06 7.5171096440058871e-06 1.7659216070078882e-06 2.3552682868282767e-05 6.0075484731317116e-06 2.9678689121826856e-05 4.5688281985000224e-06 4.2587818969459276e-06 1.2282850125910679e-05 5.6981633973611215e-06 1.2193919548692016e-05 7.7909581862542261e-06 4.1995999932004883e-06 4.2310001927379966e-06 1.4034983645177226e-06 2.2253775626039904e-06 2.5484625453534006e-06 1.5024773624760737e-06 1.1886813960082901e-06 8.943485028332714e-06 1.802211533446878e-06 8.7804607030574995e-06 1.4714171056899874e-06 3.039182778117474e-05 2.9469599285561173e-05 1.6190782721728404e-05 2.1980748656966054e-06 9.1492500963304843e-06 1.3139192984142854e-05 5.5841754669416901e-06 3.2663084979403296e-06 7.8300182408015622e-06 1.4650747681293603e-06 1.8418132244867557e-06 3.1634249051793445e-06 4.2879811205541378e-06 6.821776038991282e-06 7.2547994800721606e-06 4.5762861000866325e-06 4.0033741553487421e-06 1.3944663969273685e-05 1.5205123797572826e-06 3.7950333845819879e-05 4.6914603422440762e-06 4.3642212832058213e-06 1.6888537402380868e-05 5.3097299301474431e-06 8.5974973592752354e-06 1.0183715675617148e-05 4.4233012671049924e-06 2.8020268713604479e-06 2.4903519176724564e-06 6.0367933560789913e-06 3.6066482258866671e-06 5.5465358638439433e-06 1.6406145480373579e-05 3.6475034103942783e-06 7.545922378704344e-06 1.0510117913470496e-06 6.383917613657175e-06 4.1469930879045612e-06 1.5104979761103841e-05 7.9249357960338965e-06 6.6303162793237734e-06 2.3058946881412919e-06 6.93384276789908e-06 6.2217404008410318e-06 0.00053927751612010478 2.7688222907463807e-06 5.1593082062395665e-06 9.4327080926443393e-06 3.3336519843947502e-06 5.1198323130590842e-06 4.5094438342118166e-06 2.6237274608190453e-06 5.1693775448212788e-06 2.1082108591551617e-06 6.8329929120308474e-06 6.2018823452726071e-06 2.1240415994925091e-05 4.0243827456115514e-06 3.0522891049621393e-05 1.4011920974680818e-05 1.7239640547533074e-05 1.7993086639426091e-05 1.4355334226673438e-06 1.1012319919274514e-06 1.0614538321433708e-05 7.2890435254277739e-06 8.5872764781091643e-06 1.3084966706891505e-05 1.1094006709484758e-05 4.2456925142930984e-06 3.6872244517667462e-05 1.0859154502284048e-05 1.5319903891298572e-06 2.7727900163087534e-06 7.2483213769211959e-06 5.1159362377455894e-06 2.6822480525986132e-06 2.889767166323531e-06 3.55288675821463e-06 1.3380456162305463e-06 5.2278105015869388e-06 2.3031150972921671e-06 4.1508531796520333e-06 1.8528326040776206e-06 6.3815646996712558e-06 6.9338240811962186e-06 2.5793558575700516e-06 4.3737400474318956e-06 6.0837447954729297e-06 4.7903414469400619e-05 2.8013740155544375e-06 4.7622560053896967e-06 3.250652556381526e-06 9.5664014501971676e-06 8.2542503434926804e-07 2.5912870572853299e-06 6.0526418572379129e-06", sep=" ")
stds = np.sqrt(vars)

# DEBUG: read from file in argument instead of stdin
if len(sys.argv) > 1:
    sys.stdin = open(sys.argv[1], 'r')

def square(x):
    return x*x
square = np.vectorize(square)

# This function has to either stay in this form or implement the
# feature mapping. For details refer to the handout pdf.
def transform(x_original):
    #result_array = square(x_original)
    #result_array = np.append(x_original, result_array)
    # mean, std, #zeros

    #x_original *= 10000
    x_original -= means
    x_original /= stds
    zeros = len(x_original) - np.count_nonzero(x_original)

    result_array = x_original
    result_array = np.append(result_array, np.mean(x_original))
    result_array = np.append(result_array, np.std(x_original))
    result_array = np.append(result_array, np.min(x_original))
    result_array = np.append(result_array, np.max(x_original))
    result_array = np.append(result_array, np.median(x_original))
    result_array = np.append(result_array, np.var(x_original))
    result_array = np.append(result_array, zeros)
   # result_array = np.append(result_array, x_original[21]*x_original[21])
    result_array = np.append(result_array, 1)

    return result_array


#Parameter x: list of ndarray
#Parameter y: list of classifications
#Parameter k: corresponds to |A_t|
def process_junk(x_arr, y_arr, k, _lambda, w, t):
    if len(x_arr) == 0:
        # we have no miss classification for this junk
        return w

    zipped = zip(x_arr, y_arr)
    res = sum([tu[0] * tu[1] for tu in zipped]) / k

    # gradient
    gradient_vec = _lambda * w - res

    #learning rate
    eta = 1. / (t * _lambda)

    w -= eta * gradient_vec

    # project w back into our 1/lambda ball

    w2norm = np.linalg.norm(w, ord=2)
    w = min(1., (1. / np.sqrt(_lambda)) / w2norm) * w

    return w


# PEGASOS algorithm
# Parameter lambda - controls accuracy somehow
# Parameter k      - batch size - defines how many rows are learnt at once
#
# returns w_hat
def pegasos(_lambda, k, inpustream):
    y = []
    x = []
    t = 0

    # init w with zeros
    w = transform(np.random.rand(400))
    w = np.zeros(len(w))
    w[0] = 1/np.sqrt(_lambda)

    i = 0
    for line in inpustream:
        line = line.strip()

        i += 1

        #parse a line
        y_i = 1. if line[0:1] == '+' else -1.
        x_i = np.fromstring(line[2:], sep=" ")

        # transform feature
        x_i = transform(x_i)

        # y * <w, x> < 1
        if y_i * np.dot(w, x_i) < 0.:
            # only add miss classified samples
            x.append(x_i)
            y.append(y_i)

        if i >= k:
            t += 1
            w = process_junk(x, y, i, _lambda, w, t)

            # reset stuff
            i = 0
            y = []
            x = []

    # process last junk of data
    t += 1
    w = process_junk(x, y, i, _lambda, w, t)

    return w


if __name__ == "__main__":
    # Parameter lambda - controls accuracy somehow
    _lambda = 0.1

    # Parameter k batch size - defines how many rows are learnt at once
    k = 10

    w_hat = pegasos(_lambda, k, sys.stdin)

    w_hat_string = " ".join([repr(s) for s in w_hat])

    # use static key 1
    print '1 \t%s' % w_hat_string

