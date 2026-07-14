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
    
    vi a(n+1);
    for(int i=1;i<=n;i++) cin>>a[i];
    int cnt=0;
    set<int> s;
    for(int i=n;i>=1;i--){
        if(s.find(i)!=s.end()){
            cnt++;
        }
        else if(a[i]==i){
            cnt++;
        }
        s.insert(a[i]);
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