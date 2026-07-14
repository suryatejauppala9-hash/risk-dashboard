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

int count( string& s) {
    int cnt = 0;
    for (int i = 0; i < (int)s.length() - 1; i++) {
        if (s[i] == '(' && s[i+1] == ')') {
            cnt++;
        }
    }
    return cnt;
}

int depth(string s) {
    int k = 0;
    while (!s.empty()) {
        int bal = 0;
        bool single = true;
        
        for (int i = 0; i < s.length() - 1; i++) {
            if (s[i] == '(') bal++;
            else bal--;
            
            if (bal == 0) {
                single = false;
                break;
            }
        }
        
        if (single) {
            k++;
            s = s.substr(1, s.length() - 2);
        } else {
            break;
        }
    }
    return k;
}

void solve() {
    int n;
    cin >> n;
    
    string s,t;
    cin>>s>>t;
    vector<int> opener(n,0);
    vector<int> opener2(n,0);
    // int open=0,close=0;
    // for(int i=0;i<n;i++){
    //     if(s[i]=='(') open++;
    //     else{
    //         opener[open]++;
    //         open--;
    //     }
    // }
    // open=0;
    // for(int i=0;i<n;i++){
    //     if(s[i]=='(') open++;
    //     else{
    //         opener2[open]++;
    //         open--;
    //     }
    // }
    

    // int i=0;
    // int cnt=0;
    // while(i<n){
    //     if(s[i]!='(') {
    //         i++;
    //         continue;
    //     }
    //     while(s[i]=='('){
    //         cnt++;
    //         i++;
    //     }
    //     opener[cnt]++;
    //     cnt=0;
    // }
    // i=0;
    // cnt=0;
    // while(i<n){
    //     if(t[i]!='(') {
    //         i++;
    //         continue;
    //     }
    //     while(t[i]=='('){
    //         cnt++;
    //         i++;
    //     }
    //     opener2[cnt]++;
    //     cnt=0;
    // }
    // for(int i=0;i<n;i++){
    //     if(opener[i]!=opener2[i]) {
    //         cout<<"NO"<<endl;
    //         return;
    //     } 
    // }
    // cout<<"YES"<<endl;

    if(depth(s)==depth(t) && count(s)==count(t)){
        cout<<"YES"<<endl;
    }
    else cout<<"NO"<<endl;

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