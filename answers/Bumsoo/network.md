### TCP/IP의 각 계층을 설명해주세요.
![](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/e67b80aa-262c-45ab-ba2d-56d1b78bcf23/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210920%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210920T163153Z&X-Amz-Expires=86400&X-Amz-Signature=5d7b149dc161b379c8a0844efa4a46fc1d8f6f99726c5a0bc9b0f74f411e2380&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22)
   

### OSI 7계층와 TCP/IP 계층의 차이를 설명해주세요.
   - OSI 7계층 : 네트워크 전송 시 데이터 표준을 정리한 것으로, 개념적 모델로 통신에는 실질적으로 사용되지 않는다.
   - TCP / IP : 연결을 설정하고 네트워크를 통해 통신하는 데 실제로 사용

### Frame, Packet, Segment, Datagram을 비교해주세요.
   - Segment: If the transport protocol is TCP, the unit of data sent from TCP to network layer is called Segment.
   - Datagram: This is used in 2 layers. If the network protocol is IP, the unit of data is called Datagram. At transport layer, if protocol is UDP, we use datagram there as well. Hence, we differentiate them as UDP Datagram, IP Datagram.
   - Frame: Physical layer representation.
   - Packet: It is a more generic term used either transport layer or network layer. TCP Packet, UDP Packet, IP Packet etc. I have not seen it to represent Physical layer data units.
   - Fragment: My guess here is that when a unit of data is chopped up by a protocol to fit the MTU size, the resultant unit of data is called Fragments. But I am guessing.
   - 출처: https://stackoverflow.com/questions/11636405/definition-of-network-units-fragment-segment-packet-frame-datagram

### TCP와 UDP의 차이를 설명해주세요.
![](https://media.vlpt.us/images/taehee-kim-dev/post/409b58b9-2d04-4cb3-bfd4-d76f8472ec99/TCP,UDP%20%EC%B0%A8%EC%9D%B4%EC%A0%90.png?w=768)
   - 모두 데이터를 보내기 위해 사용하는 프로토콜
   - TCP: 인터넷상에서 데이터를 메세지의 형태로 보내기 위해 IP와 함께 사용하는 프로토콜  
   발신지와 수신지를 연결하여 패킷을 전송하기 위한 논리적 경로를 배정  
   연결형 서비스로 신뢰도를 보장하지만 속도가 느리기 때문에 파일 전송 등에 사용
   - UDP: 데이터를 데이터그램 단위로 처리하는 비연결형 프로토콜  
   비연결형 서비스로 데이터그램 방식을 제공
   UDP는 비연결형 서비스이기 때문에, 연결을 설정하고 해제하는 과정이 존재하지 않습니다.  
   서로 다른 경로로 독립적으로 처리함에도 패킷에 순서를 부여하여 재조립을 하거나 흐름 제어 또는 혼잡 제어와 같은 기능도 처리하지 않기에 TCP보다 속도가 빠르며 네트워크 부하가 적다는 장점이 있지만 신뢰성있는 데이터의 전송을 보장하지는 못합니다.  
   그렇기 때문에 신뢰성보다는 연속성이 중요한 서비스 예를 들면 실시간 서비스(streaming)에 자주 사용됩니다.
   - 참고: https://mangkyu.tistory.com/15



---
### IPv4와 IPv6 차이를 설명해주세요.
   IPv: Interner Protocol version
   IPv4: 네 도막으로 나눠진 최대 12자리의 번호
   도막마다 0~255로 2^8 가지를 표현 가능 --> 8비트
   총 4도막이니까 32비트 --> 약 40억개의 다른 주소 부여 가능

IPv6: 128비트 --> 2^128 = 1조개 이상 표현 가능
16비트씩 8부분으로 나눠서 각 부분을 :(colon)으로 구분
각 부분은 16진수로 표현(0000~ffff)
또한 서비스에 따라 각기 다른 대역폭을 확보할 수 있도록 지원, 일정한 수준의 서비스 품질(QoS)을 요구하는 실시간 서비스를 더욱 쉽게 제공할 수 있고 인증, 데이터 무결성, 데이터 기밀성을 지원하도록 보안기능을 강화
또 인터넷 주소를 기존의 「A, B, C, D」와 같은 클래스별 할당이 아닌 유니캐스트·애니캐스트·멀티캐스트 형태의 유형으로 할당하기 때문에 할당된 주소의 낭비 요인이 사라지고 더욱 간단하게 주소를 자동 설정

- 패킷을 단편화하지 않으면서도 보다 효율적인 라우팅
- 지연에 민감한 패킷을 구분하는 기본적인 QoS(Quality of Service)
- NAT를 없앰으로써 주소 공간을 32비트에서 128비트로 확장
- 네트워크 레이어 보안 내장(IPsec)
- 손쉬운 네트워크 관리를 위한 무상태 주소 자동 구성
- 처리 오버헤드가 줄어든 개선된 헤더 구조

참조: https://www.juniper.net/kr/ko/research-topics/what-is-ipv4-vs-ipv6.html

### MAC Address가 무엇인가요?
   네트워크 세그먼트의 데이터 링크 계층에서 통신을 위한 네트워크 인터페이스에 할당된 고유 식별자(위키백과)
   컴퓨터간 데이터를 전송하기 위해 있는 컴퓨터의 물리적 주소
   실제 통신에서는 ip 주소를 mac 주소로 바꿔서 사용
   mac 주소가 하드웨어 주소인 이유는 랜카드에 mac 주소가 할당돼있기 때문에

참조
https://jhnyang.tistory.com/404#index0
https://m.blog.naver.com/wood0513/222084400286