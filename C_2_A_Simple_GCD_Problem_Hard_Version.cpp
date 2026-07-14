#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using vi = vector<int>;
using vll = vector<ll>;
using pii = pair<int, int>;

#define pb push_back
#define all(x) (x).begin(), (x).end()
#define sz(x) (int)(x).size()
#define endl '\n'

const int MOD = 1e9 + 7;
const ll INF = 1e18;

void solve() {
    int n;
    cin >> n;
    
    vi a(n),b(n);
    for(int i=0;i<n;i++) cin>>a[i];
    for(int i=0;i<n;i++) cin>>b[i];
    // int gcd=1;
    // for(int i=0;i<n;i++) gcd=__gcd(gcd,a[i]);
    int cnt=0;
    // vector<int> donttouch(n,0);
    // for(int i=1;i<n;i++){
    //     if(a[i]==a[i-1]){
    //         donttouch[a[i]]=1;
    //         donttouch[a[i-1]]=1;
    //     }
    // }
    // for(int i=n-1;i>0;i--){
    //     int adj=__gcd(a[i],a[i-1]);
    //     if(adj<a[i] && !(adj==a[i-1])&&!donttouch[a[i]]) cnt++;
    // }
    // if(a[n-1]>a[n-2]) cnt++; 
    // cout<<cnt<<endl;

    // vi g;
    // for(int i=1;i<n;i++){
    //     if(donttouch[a[i]])continue;
    //     g.push_back(__gcd(a[i],a[i-1]));
    // }
    // cout<<g.size()<<endl;

    for(int i=0;i<n;i++){
        long long g1 = (i > 0) ? __gcd(a[i], a[i-1]) : 0;
        long long g2 = (i < n - 1) ? __gcd(a[i], a[i+1]) : 0;

        ll lcm=0;
        if(g1==0) lcm=g2;
        else if(g2==0) lcm=g1;
        else lcm=(g1/__gcd(g1,g2))*g2;

        // if(b[i]%lcm==0){
        //     if(b[i]==a[i]) continue;
        //     cnt++;
        // }

        if(lcm<=b[i] && lcm!=a[i]){
            cnt++;
                }
        else{
            ll v1 = (g1 > 0) ? a[i-1] / g1 : 1;
            ll v2 = (g2 > 0) ? a[i+1] / g2 : 1;

            for(int k=2;k<=40;k++){
                if((ll)k*a[i]>b[i]) break;
                if(__gcd((ll)k, v1) == 1 && __gcd((ll)k, v2) == 1) {
                    cnt++;
                    break;
                }
            }
        }
    }
    cout<<cnt<<endl;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t = 1;
    cin>>t;
    while (t--) {
        solve();
    }
    
    return 0;
}